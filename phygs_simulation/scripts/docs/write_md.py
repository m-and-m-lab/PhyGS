from glob import glob


if __name__ == '__main__':
    image_paths = glob('assets/indoor_meshes/*/*.png')

    with open('indoor_meshes.md', 'w') as f:
        f.write('# Indoor Meshes\n')
        f.write('```bash\n')
        f.write('python scripts/generate_individual_assets.py --output_folder outputs/indoor_meshes -f third_party/infinigen/tests/assets/list_indoor_meshes.txt -n 1 --save_blend --n_workers 4\n')
        f.write('```\n')
        prev_category = None
        for image_path in image_paths:
            category, subcategory = image_path.split('/')[-2:]
            subcategory = subcategory.replace('.png', '')

            if prev_category != category:
                if prev_category is not None:
                    f.write('\n')
                prev_category = category
                f.write(f'## {category}\n')
            f.write(f'### {subcategory}\n')
            f.write(f'![{subcategory}]({image_path})\n')