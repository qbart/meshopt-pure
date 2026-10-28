#!/usr/bin/env python3
"""Generates the synthetic PNG edge cases and tex.gltf (a quad per texture) used by the BasisU golden cases."""
import base64, json, os, struct, zlib

DIR = os.path.dirname(os.path.abspath(__file__))


def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def pack_row(samples, bits):
    if bits >= 8:
        return b''.join(s.to_bytes(bits // 8, 'big') for s in samples)
    out, acc, n = bytearray(), 0, 0
    for s in samples:
        acc = (acc << bits) | s
        n += bits
        if n == 8:
            out.append(acc)
            acc, n = 0, 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


def png(name, w, h, color_type, bits, pixel, extra=b'', interlace=0):
    """pixel(x, y) returns the channel samples for that pixel."""
    raw = bytearray()
    passes = ADAM7 if interlace else [(0, 0, 1, 1)]
    for (x0, y0, dx, dy) in passes:
        xs = range(x0, w, dx)
        if not len(xs):
            continue
        for y in range(y0, h, dy):
            row = [s for x in xs for s in pixel(x, y)]
            raw += b'\0' + pack_row(row, bits)
    ihdr = struct.pack('>IIBBBBB', w, h, bits, color_type, 0, 0, interlace)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + extra + chunk(b'IDAT', zlib.compress(bytes(raw), 9)) + chunk(b'IEND', b'')
    with open(os.path.join(DIR, name), 'wb') as f:
        f.write(data)
    return name


def grad(x, y, w, h, m):
    return (x * m) // max(w - 1, 1), (y * m) // max(h - 1, 1), ((x + y) * m) // max(w + h - 2, 1)


images = [
    png('rgb16.png', 13, 9, 2, 16, lambda x, y: grad(x, y, 13, 9, 65535)),
    png('rgba16.png', 8, 8, 6, 16, lambda x, y: grad(x, y, 8, 8, 65535) + ((x * 9000) % 65536,)),
    png('gray1.png', 17, 3, 0, 1, lambda x, y: ((x + y) & 1,)),
    png('gray8trns.png', 6, 6, 0, 8, lambda x, y: ((x * 40 + y) % 256,), chunk(b'tRNS', struct.pack('>H', 40))),
    png('graya16.png', 5, 7, 4, 16, lambda x, y: (x * 10000, y * 9000)),
    png('pal2trns.png', 9, 5, 3, 2, lambda x, y: ((x + 2 * y) % 4,),
        chunk(b'PLTE', bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 250, 250, 10])) + chunk(b'tRNS', bytes([255, 128, 0]))),
    png('rgbtrns.png', 4, 4, 2, 8, lambda x, y: (x * 60, y * 60, 0), chunk(b'tRNS', struct.pack('>HHH', 60, 60, 0))),
    png('rgbai.png', 11, 10, 6, 8, lambda x, y: grad(x, y, 11, 10, 255) + (255 - x * 20,), interlace=1),
    png('tall.png', 1, 9, 2, 8, lambda x, y: (y * 28, 255 - y * 28, 128)),
    png('wide.png', 37, 2, 6, 8, lambda x, y: (x * 7, y * 100, 50, 255)),
]

# a unit quad per texture; textures alternate between color, normal and occlusion use
positions = struct.pack('<12f', 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0)
uvs = struct.pack('<8f', 0, 1, 1, 1, 1, 0, 0, 0)
indices = struct.pack('<6H', 0, 1, 2, 0, 2, 3)
buf = positions + uvs + indices

materials, meshes, nodes = [], [], []
for i, _ in enumerate(images):
    if i % 3 == 0:
        materials.append({'pbrMetallicRoughness': {'baseColorTexture': {'index': i}}})
    elif i % 3 == 1:
        materials.append({'normalTexture': {'index': i}})
    else:
        materials.append({'occlusionTexture': {'index': i}, 'alphaMode': 'BLEND'})
    meshes.append({'primitives': [{'attributes': {'POSITION': 0, 'TEXCOORD_0': 1}, 'indices': 2, 'material': i}]})
    nodes.append({'mesh': i, 'translation': [i * 1.5, 0, 0]})

gltf = {
    'asset': {'version': '2.0'},
    'scene': 0,
    'scenes': [{'nodes': list(range(len(nodes)))}],
    'nodes': nodes,
    'meshes': meshes,
    'materials': materials,
    'textures': [{'source': i} for i in range(len(images))],
    'images': [{'uri': n} for n in images],
    'buffers': [{'byteLength': len(buf), 'uri': 'data:application/octet-stream;base64,' + base64.b64encode(buf).decode()}],
    'bufferViews': [
        {'buffer': 0, 'byteOffset': 0, 'byteLength': 48},
        {'buffer': 0, 'byteOffset': 48, 'byteLength': 32},
        {'buffer': 0, 'byteOffset': 80, 'byteLength': 12},
    ],
    'accessors': [
        {'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3', 'min': [0, 0, 0], 'max': [1, 1, 0]},
        {'bufferView': 1, 'componentType': 5126, 'count': 4, 'type': 'VEC2'},
        {'bufferView': 2, 'componentType': 5123, 'count': 6, 'type': 'SCALAR'},
    ],
}

with open(os.path.join(DIR, 'tex.gltf'), 'w') as f:
    json.dump(gltf, f, indent=1)
