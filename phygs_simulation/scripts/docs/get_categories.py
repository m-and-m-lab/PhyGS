from glob import glob
import importlib
from inspect import getmodule
import infinigen.assets.objects
import os


if __name__ == '__main__':
    factories_category = {}
    for path in glob('third_party/infinigen/infinigen/assets/objects/*'):
        category = os.path.basename(path)
        category_factories = importlib.import_module(
            '.' + category,
            'infinigen.assets.objects',
        )

        for name, factory in category_factories.__dict__.items():
            if not name.endswith('Factory'):
                continue

            factories_category[factory.__name__] = category


        # print(category, filename, factory.__name__)
        # print(path)
        # print(getmodule(infinigen.assets.objects, ))

    print(factories_category)