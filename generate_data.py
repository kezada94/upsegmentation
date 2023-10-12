import random
import shutil
import requests
from io import BytesIO
from pathlib import Path

import numpy as np

import svgutils
from PIL import Image, ImageOps
from cairosvg import svg2png


def download_and_unzip(url: str, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # download the zip
    r = requests.get(url, allow_redirects=True)
    zip_path.open('wb').write(r.content)

    # un zipit
    shutil.unpack_archive(zip_path, zip_path.with_suffix(''))


def download_zip(path, download_link, download_name):
    zip_path = path / download_name

    if not zip_path.with_suffix('').exists():
        download_and_unzip(download_link, zip_path)


def generate_svg(src_path, dst_path, num_images):
    dst_path.mkdir(parents=True, exist_ok=True)

    images = list(src_path.glob('*.svg'))

    for i in range(num_images):
        elements = []
        for svg_file in random.sample(images, random.randint(5, 10)):
            svg = svgutils.compose.SVG(str(svg_file))
            svg.scale(1.0 / (max(svg.width, svg.height)))
            svg.scale(random.uniform(0.5, 1.5))
            svg.rotate(random.randint(0, 360))
            svg.move(random.uniform(0, 1.5), random.uniform(0, 1.5))
            svg.scale(256)
            elements.append(svg)
        figure = svgutils.compose.Figure(512, 512, *elements)
        figure.save(str(dst_path / f'pic_{i:04d}.svg'))


def get_image(file: Path, width, height):
    with file.open('r') as f:
        return Image.open(BytesIO(svg2png(file_obj=f, output_width=width, output_height=height)))


def raster_svg(src_path, dst_path, input_width, input_height, output_width, output_height, texture_path = None):
    dst_path.mkdir(parents=True, exist_ok=True)

    textures = []

    if texture_path is not None:
        for file in texture_path.glob('*'):
            try:
                textures.append(Image.open(str(file)))
            except Exception as e:
                print(e)

    for file in src_path.glob('*.svg'):
        # target
        image = get_image(file, output_width, output_height)
        (Image
         .frombuffer("L", image.size, image.tobytes("raw", "A"))
         .save(str(dst_path / f"{file.stem}_target.png")))

        # input
        image = get_image(file, input_width, input_height)
        image = np.asarray(image)

        # pick a random texture
        random_texture = random.choice(textures).copy()
        random_texture = random_texture.transform(random_texture.size,
                                                  Image.AFFINE,
                                                  (1, 0, random.uniform(-0.5, 0.5), 0, 1, random.uniform(-0.5, 0.5)))
        random_texture = np.asarray(random_texture)

        texture_height, texture_width = random_texture.shape[:2]

        tile_x = int(np.ceil(input_width / texture_width))
        tile_y = int(np.ceil(input_height / texture_height))

        random_texture = np.tile(random_texture, (tile_x, tile_y))
        random_texture = random_texture[:input_height, :input_width]

        # use image alpha to blend with random texture
        alpha = image[:, :, 3, np.newaxis] / 255
        image = image[:, :, :3]

        image = (image * alpha + random_texture * (1 - alpha)).astype(np.uint8)

        image = Image.frombuffer('RGB', (input_width, input_height), image, 'raw', 'RGB', 0, 1)
        image = ImageOps.grayscale(image)
        image.save(str(dst_path / file.with_suffix('.png').name))


def main():
    path = Path('data')
    num_images = 1000

    # download_zip(path, "https://github.com/jnovack/pokemon-svg/archive/refs/heads/master.zip", "pokemon-svg.zip")
    # download_zip(path, "https://opengameart.org/sites/default/files/seamlessTextures2.zip", "textures.zip")
    # generate_svg(path / 'pokemon-svg' / 'pokemon-svg-master' / 'svg', path / 'generated' / 'svg', num_images)
    raster_svg(path / 'generated' / 'svg', path / 'generated' / 'png', 112, 112, 186, 186, path / 'textures')


if __name__ == "__main__":
    main()
