#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import getpass
import logging
import mapnik
import multiprocessing
import os
import sys
from math import pi, sin, log, exp, atan
from shapely.geometry import Polygon
from shapely.wkb import loads

try:
    import psycopg2
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

DEG_TO_RAD = pi / 180
RAD_TO_DEG = 180 / pi
TILE_SIZE = 256
LIST_QUEUE_LENGTH = 32


def box(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

def minmax(a, b, c):
    a = max(a, b)
    a = min(a, c)
    return a

def format2ext(format):
    ext = "png"
    if format.startswith("jpeg"):
        ext = "jpg"
    elif format.startswith("tif"):
        ext = "tif"
    elif format.startswith("webp"):
        ext = "webp"
    return ext


class GoogleProjection:
    def __init__(self, levels=22):
        self.Bc = []
        self.Cc = []
        self.zc = []
        self.Ac = []
        c = 256
        for d in range(0, levels):
            e = c / 2
            self.Bc.append(c / 360.0)
            self.Cc.append(c / (2 * pi))
            self.zc.append((e, e))
            self.Ac.append(c)
            c *= 2

    def fromLLtoPixel(self, ll, zoom):
        d = self.zc[zoom]
        e = round(d[0] + ll[0] * self.Bc[zoom])
        f = minmax(sin(DEG_TO_RAD * ll[1]), -0.9999, 0.9999)
        g = round(d[1] + 0.5 * log((1 + f) / (1 - f)) * -self.Cc[zoom])
        return (e, g)

    def fromPixelToLL(self, px, zoom):
        e = self.zc[zoom]
        f = (px[0] - e[0]) / self.Bc[zoom]
        g = (px[1] - e[1]) / -self.Cc[zoom]
        h = RAD_TO_DEG * (2 * atan(exp(g)) - 0.5 * pi)
        return (f, h)


class ListWriter:
    def __init__(self, f):
        self.f = f

    def __str__(self):
        return "ListWriter({0})".format(self.f.name)

    def write_poly(self, poly):
        self.f.write("BBox: {0}\n".format(poly.bounds))

    def write(self, x, y, z):
        self.f.write("{0} {1} {2}\n".format(x, y, z))

    def exists(self, x, y, z):
        return False

    def need_image(self):
        return False

    def multithreading(self):
        return False

    def close(self):
        self.f.close()


class FileWriter:
    def __init__(self, tile_dir, format="png256", tms=False, overwrite=True):
        self.format = format
        self.overwrite = overwrite
        self.tms = tms
        self.tile_dir = tile_dir
        if not self.tile_dir.endswith("/"):
            self.tile_dir = self.tile_dir + "/"
        if not os.path.isdir(self.tile_dir):
            os.mkdir(self.tile_dir)

    def __str__(self):
        return "FileWriter({0})".format(self.tile_dir)

    def write_poly(self, poly):
        pass

    def tile_uri(self, x, y, z):
        return "{0}{1}/{2}/{3}.{4}".format(self.tile_dir, z, x, y if not self.tms else 2**z - 1 - y, format2ext(self.format))

    def exists(self, x, y, z):
        return os.path.isfile(self.tile_uri(x, y, z))

    def write(self, x, y, z, image):
        uri = self.tile_uri(x, y, z)
        try:
            os.makedirs(os.path.dirname(uri))
        except OSError:
            pass
        if self.overwrite or not os.path.exists(uri):
            image.save(uri, self.format)

    def need_image(self):
        return True

    def multithreading(self):
        return True

    def close(self):
        pass


# https://github.com/mapbox/mbutil/blob/master/mbutil/util.py
class MBTilesWriter:
    def __init__(self, filename, setname, overlay=False, version=1, description=None, format="png256"):
        self.format = format
        self.filename = filename
        if not self.filename.endswith(".mbtiles"):
            self.filename = self.filename + ".mbtiles"
        self.con = sqlite3.connect(self.filename, isolation_level=None)
        self.cur = self.con.cursor()
        self.cur.execute("PRAGMA synchronous=0")
        self.cur.execute("PRAGMA locking_mode=EXCLUSIVE")
      # self.cur.execute('PRAGMA journal_mode=TRUNCATE')
        self.cur.execute("CREATE TABLE IF NOT EXISTS tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);")
        self.cur.execute("CREATE TABLE IF NOT EXISTS metadata (name text, value text);")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS name ON metadata (name);")
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS tile_index ON tiles (zoom_level, tile_column, tile_row);")

        metadata = [("name", setname), ("format", format2ext(self.format)), ("type", "overlay" if overlay else "baselayer"), ("version", version)]
        if description:
            metadata.append(("description", description))
        self.cur.executemany("REPLACE INTO metadata (name, value) VALUES (?, ?)", metadata)

    def __str__(self):
        return "MBTilesWriter({0})".format(self.filename)

    def write_poly(self, poly):
        bbox = poly.bounds
        self.cur.execute("SELECT value FROM metadata WHERE name='bounds'")
        result = self.cur.fetchone
        if result:
            b = result["value"].split(",")
            oldbbox = box(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            bbox = bbox.union(oldbbox).bounds
        self.cur.execute("REPLACE INTO metadata (name, value) VALUES ('bounds', ?)", ",".join(bbox))

    def exists(self, x, y, z):
        query = "SELECT 1 FROM tiles WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?"
        self.cur.execute(query, (z, x, 2**z - 1 - y))
        return self.cur.fetchone()

    def write(self, x, y, z, image):
        query = "REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) values (?, ?, ?, ?);"
        self.cur.execute(query, (z, x, 2**z - 1 - y, sqlite3.Binary(image.tostring(self.format))))
        self.con.commit()

    def need_image(self):
        return True

    def multithreading(self):
        return False

    def close(self):
        self.con.commit()
        self.cur.execute("ANALYZE;")
        self.cur.execute("VACUUM;")
        self.cur.close()
        self.con.close()


class FakeImage:
    def __init__(self, image, format):
        self.imgstr = image.tostring(format)

    def tostring(self, format):
        return self.imgstr

    def save(self, uri, format):
        with open(uri, "wb") as f:
            f.write(self.imgstr)


class ThreadedWriterWrapper(multiprocessing.Process):
    def __init__(self, wclass, wparams):
        super(ThreadedWriterWrapper, self).__init__()
        self.wclass = wclass
        self.wparams = wparams
        self.format = wparams.get("format", "png256")
        self.queue = multiprocessing.Queue(maxsize=100)
        self.daemon = True
        self.start()
        self.ni, self.desc = self.queue.get(block=True, timeout=15)

    def __str__(self):
        return "Threaded{}".format(self.desc)

    def run(self):
        writerConstr = globals().get(self.wclass)
        if not writerConstr:
            return
        writer = writerConstr(**self.wparams)
        need_image = writer.need_image()
        self.queue.put((need_image, str(writer)))
        while True:
            req, args = self.queue.get()
            if req == "close":
                break
            elif req == "write_poly":
                writer.write_poly(args)
            elif req == "write":
                if need_image:
                    writer.write(args[0], args[1], args[2], args[3])
                else:
                    writer.write(args[0], args[1], args[2])
        writer.close()

    def write_poly(self, poly):
        self.queue.put(("write_poly", poly))

    def exists(self, x, y, z):
        return False

    def write(self, x, y, z, image):
        self.queue.put(("write", (x, y, z, FakeImage(image, self.format))))

    def need_image(self):
        return self.ni

    def multithreading(self):
        return True

    def close(self):
        self.queue.put(("close", None))
        super().join()


def multi_MBTilesWriter(threads, filename, setname, overlay=False, version=1, description=None, format="png256"):
    params = {"filename": filename, "setname": setname, "overlay": overlay, "version": version, "description": description, "format": format}
    if threads == 1:
        return MBTilesWriter(**params)
    else:
        return ThreadedWriterWrapper("MBTilesWriter", params)


class RenderTask:
    def __init__(self, metatile, zoom, x, y):
        self.zoom = zoom
        self.metatile = metatile
        self.mtx0 = x - x % metatile
        self.mty0 = y - y % metatile
        self._tiles = set()
        # projects between tile pixel coordinates and LatLong (EPSG:4326)
        self.tileproj = GoogleProjection()

    def add(self, x, y):
        if self.belongs(x, y):
            self._tiles.add((x, y))

    def belongs(self, x, y, z=-1):
        return (z < 0 or z == self.zoom) and x >= self.mtx0 and y >= self.mty0 and x < self.mtx0 + self.metatile and y < self.mty0 + self.metatile

    def tiles(self):
        for t in self._tiles:
            yield (t[0], t[1], self.zoom, t[0] - self.mtx0, t[1] - self.mty0)

    def get_bbox(self):
        # calculate pixel positions of bottom-left & top-right
        p0 = (self.mtx0 * TILE_SIZE, (self.mty0 + self.metatile) * TILE_SIZE)
        p1 = ((self.mtx0 + self.metatile) * TILE_SIZE, self.mty0 * TILE_SIZE)

        # convert to LatLong (EPSG:4326)
        l0 = self.tileproj.fromPixelToLL(p0, self.zoom)
        l1 = self.tileproj.fromPixelToLL(p1, self.zoom)

        # convert to map projection (e.g. mercator coordinates EPSG:900913)
        c0 = self.prj.forward(mapnik.Coord(l0[0], l0[1]))
        c1 = self.prj.forward(mapnik.Coord(l1[0], l1[1]))

        # bounding box for the tile
        if hasattr(mapnik, "mapnik_version") and mapnik.mapnik_version() >= 800:
            bbox = mapnik.Box2d(c0.x, c0.y, c1.x, c1.y)
        else:
            bbox = mapnik.Envelope(c0.x, c0.y, c1.x, c1.y)
        return bbox


class RenderThread:
    def __init__(self, writer, maprenderer, q, printLock, scale=1.0, renderlist=False):
        self.writer = writer
        self.q = q
        self.printLock = printLock
        self.renderlist = renderlist
        self.m = maprenderer.get_map()
        self.p = maprenderer.get_projection()
        self.scale = maprenderer.get_scale()
        self.scaled_size = maprenderer.get_scaled_size()

    def render_task(self, task):
        bbox = task.get_bbox()
        render_size = task.metatile * self.scaled_size
        self.m.resize(render_size, render_size)
        self.m.zoom_to_box(bbox)
        self.m.buffer_size = int(self.scaled_size / 2)

        # render image with default AGG renderer
        im = mapnik.Image(render_size, render_size)
        mapnik.render(self.m, im, self.scale)

        # now cut parts of the image to tiles
        for t in task.tiles():
            if task.metatile == 1:
                self.writer.write(t[0], t[1], t[2], im)
            else:
                self.writer.write(t[0], t[1], t[2], im.view(t[3] * self.scaled_size, t[4] * self.scaled_size, self.scaled_size, self.scaled_size))
            logger = logging.getLogger(__name__)
            if logger.isEnabledFor(logging.INFO):
                with self.printLock:
                    logger.info(f"{t[2]} {t[0]} {t[1]}")

    def loop(self):
        while True:
            # fetch a tile from the queue and render it
            task = self.q.get()
            if task is None:
                self.q.task_done()
                break

            if self.writer.need_image():
                task.prj = self.p
                self.render_task(task)
            else:
                for t in task.tiles():
                    self.writer.write(t[0], t[1], t[2])
                    if self.renderlist:
                        break
            self.q.task_done()


class ListGenerator:
    def __init__(self, f, metatile=1):
        self.f = f
        self.metatile = metatile

    def __str__(self):
        return "ListGenerator({0})".format(self.f.name)

    def generate(self, queue):
        import re

        metatiles = []
        for line in self.f:
            m = re.search(r"(\d+)\D+(\d+)\D+(\d{1,2})", line)
            if m:
                x = int(m.group(1))
                y = int(m.group(2))
                z = int(m.group(3))
                if self.metatile == 1:
                    queue.put(RenderTask(1, z, x, y))
                else:
                    n = -1
                    for i in range(len(metatiles)):
                        if metatiles[i].belongs(x, y, z=z):
                            n = i
                            break
                    if n >= 0:
                        m = metatiles.pop(n)
                    else:
                        m = RenderTask(self.metatile, z, x, y)
                        while len(metatiles) >= LIST_QUEUE_LENGTH:
                            queue.put(metatiles.pop(0))
                    m.add(x, y)
                    metatiles.append(m)
        for m in metatiles:
            queue.put(m)


class PolyGenerator:
    def __init__(self, poly, zooms, metatile=1):
        self.poly = poly
        self.zooms = zooms
        self.zooms.sort()
        self.metatile = metatile

    def __str__(self):
        return "PolyGenerator({0}, {1})".format(self.poly.bounds, self.zooms)

    def check_tile(self, x, y, z):
        # calculate pixel positions of bottom-left & top-right
        tt_p0 = (x * TILE_SIZE, (y + 1) * TILE_SIZE)
        tt_p1 = ((x + 1) * TILE_SIZE, y * TILE_SIZE)

        # convert to LatLong (EPSG:4326)
        tt_l0 = self.gprj.fromPixelToLL(tt_p0, z)
        tt_l1 = self.gprj.fromPixelToLL(tt_p1, z)

        tt_p = box(tt_l0[0], tt_l1[1], tt_l1[0], tt_l0[1])
        return self.poly.intersects(tt_p)

    def generate(self, queue):
        self.gprj = GoogleProjection(self.zooms[-1] + 1)

        bbox = self.poly.bounds
        ll0 = (bbox[0], bbox[3])
        ll1 = (bbox[2], bbox[1])

        for z in self.zooms:
            px0 = self.gprj.fromLLtoPixel(ll0, z)
            px1 = self.gprj.fromLLtoPixel(ll1, z)

            xmin = max(0, min(2**z - 1, int(px0[0] / float(TILE_SIZE))))
            xmax = max(0, min(2**z - 1, int(px1[0] / float(TILE_SIZE))))
            ymin = max(0, min(2**z - 1, int(px0[1] / float(TILE_SIZE))))
            ymax = max(0, min(2**z - 1, int(px1[1] / float(TILE_SIZE))))

            x0_min = xmin - xmin % self.metatile
            y0_min = ymin - ymin % self.metatile
            x0_max = xmax - xmax % self.metatile
            y0_max = ymax - ymax % self.metatile

            for x0 in range(x0_min, x0_max + 1, self.metatile):
                for y0 in range(y0_min, y0_max + 1, self.metatile):
                    t = None
                    for x in range(x0, x0 + self.metatile):
                        for y in range(y0, y0 + self.metatile):
                            if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                                if self.check_tile(x, y, z):
                                    if not t:
                                        t = RenderTask(self.metatile, z, x0, y0)
                                    t.add(x, y)
                    if t:
                        queue.put(t)


class MapnikRenderer:
    def __init__(self, mapfile, scale):
        self.mapfile = mapfile
        self.scale = scale
        self.scaled_size = int(TILE_SIZE * scale)
        self.m = mapnik.Map(self.scaled_size, self.scaled_size)
        mapnik.load_map(self.m, self.mapfile, True)
        self.prj = mapnik.Projection(self.m.srs)

    def get_map(self):
        return self.m

    def get_mapfile(self):
        return self.mapfile

    def get_projection(self):
        return self.prj

    def get_scale(self):
        return self.scale

    def get_scaled_size(self):
        return self.scaled_size


def render_tiles_multithreaded(generator, maprenderer, writer, num_threads=2, scale=1.0, renderlist=False):
    logging.info("render_tiles_multithreaded(%s %s %s %s)", generator, maprenderer.get_mapfile(), writer, num_threads)
    printLock = multiprocessing.Lock()
    queue = multiprocessing.JoinableQueue(32)

    renderers = {}
    for i in range(num_threads):
        renderer = RenderThread(writer, maprenderer, queue, printLock, scale=scale, renderlist=renderlist)
        render_thread = multiprocessing.Process(target=renderer.loop)
        render_thread.start()
        renderers[i] = render_thread

    generator.generate(queue)

    # signal render threads to exit by sending empty request to queue
    for i in range(num_threads):
        queue.put(None)

    # wait for pending rendering jobs to complete
    queue.join()
    for i in range(num_threads):
        renderers[i].join()


def render_tiles(generator, maprenderer, writer, num_threads=1, scale=1.0, renderlist=False):
    logging.info("render_tiles(%s %s %s)", generator, maprenderer.get_mapfile(), writer)
    printLock = multiprocessing.Lock()

    queue = multiprocessing.JoinableQueue(0)
    generator.generate(queue)

    renderer = RenderThread(writer, maprenderer, queue, printLock, scale=scale, renderlist=renderlist)
    queue.put(None)
    renderer.loop()


def poly_parse(fp):
    result = None
    poly = []
    data = False
    hole = False
    for l in fp:
        l = l.strip()
        if l == "END" and data:
            if len(poly) > 0:
                if hole and result:
                    result = result.difference(Polygon(poly))
                elif not hole and result:
                    result = result.union(Polygon(poly))
                elif not hole:
                    result = Polygon(poly)
            poly = []
            data = False
        elif l == "END" and not data:
            break
        elif len(l) > 0 and " " not in l and "\t" not in l:
            data = True
            hole = l[0] == "!"
        elif l and data:
            poly.append([float(x.strip()) for x in l.split()[:2]])
    return result


def read_db(db, osm_id=0):
    # zero for DB BBOX
    cur = db.cursor()
    if osm_id:
        cur.execute("SELECT ST_Transform(way, 4326) FROM planet_osm_polygon WHERE osm_id = %s;", (osm_id,))
    else:
        cur.execute("SELECT ST_Transform(ST_ConvexHull(ST_Collect(way)), 4326) FROM planet_osm_polygon;")
    way = cur.fetchone()[0]
    cur.close()
    return loads(way.decode("hex"))


def read_cities(db, osm_id=0):
    cur = db.cursor()
    if osm_id:
        cur.execute("SELECT ST_Transform(ST_Union(pl.way), 4326) FROM planet_osm_polygon pl, planet_osm_polygon b WHERE b.osm_id = %s AND pl.place IN ('town', 'city') AND ST_Area(pl.way) < 500*1000*1000 AND ST_Contains(b.way, pl.way);", (osm_id,))
    else:
        cur.execute("SELECT ST_Transform(ST_Union(way), 4326) FROM planet_osm_polygon WHERE place IN ('town', 'city') AND ST_Area(way) < 500*1000*1000;")
    result = cur.fetchone()
    poly = loads(result[0].decode("hex")) if result else Polygon()
    if osm_id:
        cur.execute("SELECT ST_Transform(ST_Union(ST_Buffer(p.way, 5000)), 4326) FROM planet_osm_point p, planet_osm_polygon b WHERE b.osm_id=%s AND ST_Contains(b.way, p.way) AND p.place IN ('town', 'city') AND NOT EXISTS(SELECT 1 FROM planet_osm_polygon pp WHERE pp.name=p.name AND ST_Contains(pp.way, p.way));", (osm_id,))
    else:
        cur.execute("SELECT ST_Transform(ST_Union(ST_Buffer(p.way, 5000)), 4326) FROM planet_osm_point p WHERE p.place in ('town', 'city') AND NOT EXISTS(SELECT 1 FROM planet_osm_polygon pp WHERE pp.name=p.name AND ST_Contains(pp.way, p.way));")
    result = cur.fetchone()
    if result:
        poly = poly.union(loads(result[0].decode("hex")))
    return poly


def main():
    try:
        mapfile = os.environ["MAPNIK_MAP_FILE"]
    except KeyError:
        mapfile = os.getcwd() + "/osm.xml"

    default_user = getpass.getuser()

    parser = argparse.ArgumentParser(description="Generate mapnik tiles for an OSM polygon")
    apg_input = parser.add_argument_group("Input")
    apg_input.add_argument("-b", "--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"), help="generate tiles inside a bounding box")
    apg_input.add_argument("-p", "--poly", type=argparse.FileType("r"), help="use a poly file for area")
    if HAS_PSYCOPG:
        apg_input.add_argument("-a", "--area", type=int, metavar="OSM_ID", help="generate tiles inside an OSM polygon: positive for polygons, negative for relations, 0 for whole database")
        apg_input.add_argument("-c", "--cities", type=int, metavar="OSM_ID", help="generate tiles for all towns inside a polygon")
    apg_input.add_argument("-l", "--list", type=argparse.FileType("r"), metavar="TILES.LST", help="process tile list")
    apg_output = parser.add_argument_group("Output")
    apg_output.add_argument("-t", "--tiledir", metavar="DIR", help="output tiles to directory (default: {0}/tiles)".format(os.getcwd()))
    apg_output.add_argument("--tms", action="store_true", help="write files in TMS order", default=False)
    if HAS_SQLITE:
        apg_output.add_argument("-m", "--mbtiles", help="generate mbtiles file")
        apg_output.add_argument("--name", help="name for mbtiles", default="Test MBTiles")
        apg_output.add_argument("--overlay", action="store_true", help="if this layer is an overlay (for mbtiles metadata)", default=False)
    apg_output.add_argument("-x", "--export", type=argparse.FileType("w"), metavar="TILES.LST", help="save tile list into file")
    apg_output.add_argument("-z", "--zooms", type=int, nargs=2, metavar=("ZMIN", "ZMAX"), help="range of zoom levels to render (default: 6 12)", default=(6, 12))
    apg_other = parser.add_argument_group("Settings")
    apg_other.add_argument("-s", "--style", help="style file for mapnik (default: {0})".format(mapfile), default=mapfile)
    apg_other.add_argument("-f", "--format", default="png256", help="tile image format (default: png256)")
    apg_other.add_argument("--meta", type=int, default=8, metavar="N", help="metatile size NxN tiles (default: 8)")
    apg_other.add_argument("--scale", type=float, default=1.0, help="scale factor for HiDpi tiles (affects tile size)")
    apg_other.add_argument("--threads", type=int, metavar="N", help="number of threads (default: 2)", default=2)
    apg_other.add_argument("--skip-existing", action="store_true", default=False, help="do not overwrite existing files")
    apg_other.add_argument("--fonts", help="directory with custom fonts for the style")
    apg_other.add_argument("--for-renderd", action="store_true", default=False, help="produce only a single tile for metatiles")
    apg_other.add_argument("-q", "--quiet", dest="verbose", action="store_false", help="do not print any information", default=True)
    if HAS_PSYCOPG:
        apg_db = parser.add_argument_group("Database (for poly/cities)")
        apg_db.add_argument("-d", "--dbname", metavar="DB", help="database (default: gis)", default="gis")
        apg_db.add_argument("--host", help="database host", default=None)
        apg_db.add_argument("--port", type=int, help="database port", default="5432")
        apg_db.add_argument("-u", "--user", help="user name for db (default: {0})".format(default_user), default=default_user)
        apg_db.add_argument("-w", "--password", action="store_true", help="ask for password", default=False)
    options = parser.parse_args()

    # check for required argument
    if not options.bbox and not options.poly and (not HAS_PSYCOPG or (not options.cities and not options.area)) and not options.list:
        parser.print_help()
        sys.exit(1)

    if options.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # custom fonts
    if options.fonts:
        mapnik.register_fonts(options.fonts)

    # writer
    if options.tiledir:
        writer = FileWriter(options.tiledir, format=options.format, tms=options.tms, overwrite=not options.skip_existing)
    elif HAS_SQLITE and options.mbtiles:
        writer = multi_MBTilesWriter(options.threads, options.mbtiles, options.name, overlay=options.overlay, format=options.format)
    elif options.export:
        writer = ListWriter(options.export)
    else:
        writer = FileWriter(os.getcwd() + "/tiles", format=options.format, tms=options.tms, overwrite=not options.skip_existing)

    # input and process
    poly = None
    if options.bbox:
        b = options.bbox
        poly = box(b[0], b[1], b[2], b[3])
    if options.poly:
        tpoly = poly_parse(options.poly)
        poly = tpoly if not poly else poly.intersection(tpoly)
    if HAS_PSYCOPG and (options.area or options.cities):
        passwd = None
        if options.password:
            passwd = getpass.getpass("Please enter your password: ")
        try:
            db = psycopg2.connect(database=options.dbname, user=options.user, password=passwd, host=options.host, port=options.port)
            if options.area:
                tpoly = read_db(db, options.area)
                poly = tpoly if not poly else poly.intersection(tpoly)
            if options.cities:
                tpoly = read_cities(db, options.cities)
                poly = tpoly if not poly else poly.intersection(tpoly)
            db.close()
        except Exception as e:
            logging.error("Error connecting to database: %s", e.pgerror or e)
            sys.exit(3)

    if options.list:
        generator = ListGenerator(options.list, metatile=options.meta)
    elif poly:
        generator = PolyGenerator(poly, list(range(options.zooms[0], options.zooms[1] + 1)), metatile=options.meta)
    else:
        logging.error("Please specify a region for rendering.")
        sys.exit(4)

    renderer = MapnikRenderer(options.style, options.scale)
    if options.threads > 1 and writer.multithreading():
        render_tiles_multithreaded(generator, renderer, writer, num_threads=options.threads, renderlist=options.for_renderd)
    else:
        render_tiles(generator, renderer, writer, renderlist=options.for_renderd)

    writer.close()


if __name__ == "__main__":
    main()
