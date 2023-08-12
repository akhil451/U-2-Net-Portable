"""
The code uses the gooey library to create a GUI for the script. It allows users to select image sources ("pexels" and/or "bing"), enter categories to download, specify the number of images to download per class, and choose an output location.

The main function is decorated with @Gooey to create the GUI interface. It defines the command-line arguments using GooeyParser and processes the selected options. The args object contains the parsed arguments.

The code checks if the output location directory exists and creates it if it doesn't. It then proceeds to download images from Pexels or Bing based on the selected sources. The download_pexel_images function is called if "pexels" is selected, and the bing_downloader function is called if "bing" is selected.

Finally, the main function is called if the script is run directly.
"""

import os
import sys
from gooey import Gooey, GooeyParser
# from pexels.pexels import download_pexel_images
# from pixabay.pixabay import download_pixabay_images
# from bing.bing import bing_downloader
# from settings import pexels_auth_code,pixabay_api_code
from charset_normalizer import md__mypyc
# from saliency import main    
from u2net_test import process_saliency
@Gooey()
def main():
    parser = GooeyParser(description="CREATE IMAGE DATASET")
    # parser.add_argument(
    #     "SOURCES",
    #     metavar="SOURCES",
    #     help="SOURCES",
    #     widget="Listbox",
    #     nargs="+",
    #     choices=["Pexels", "Bing","Pixabay"],
    # )
    # parser.add_argument(
    #     "CATEGORIES", type=str, help="Enter categories to download, separated by comma"
    # )
    # parser.add_argument(
    #     "IMAGES_PER_CLASS",
    #     type=str,
    #     help="Enter number of images to download per class",
    # )
    parser.add_argument("INPUT_LOCATION", help="Choose the input location of Images", widget="DirChooser")
    parser.add_argument("OUTPUT_LOCATION", help="Choose the location to save images", widget="DirChooser")
    args = parser.parse_args()
    # categories = args.CATEGORIES.strip(" ").split(",")
    # images_per_class =  int(args.IMAGES_PER_CLASS)//len(args.SOURCES)
    # images_per_class =  int(args.IMAGES_PER_CLASS)

    if not os.path.exists(args.OUTPUT_LOCATION):
        os.makedirs(args.OUTPUT_LOCATION)

    process_saliency(args.INPUT_LOCATION,args.OUTPUT_LOCATION)
    # if "Pexels" in args.SOURCES:
    #     download_pexel_images(
    #         auth_code=pexels_auth_code,
    #         output_loc=args.OUTPUT_LOCATION,
    #         n_images_per_class=images_per_class,
    #         categories=categories,
    #     )
    # if "Bing" in args.SOURCES:
    # bing_downloader(
    #     categories,
    #     n_images_per_category=int(images_per_class),
    #     output_dir=args.OUTPUT_LOCATION,
    # )




if __name__ == "__main__":
    main()