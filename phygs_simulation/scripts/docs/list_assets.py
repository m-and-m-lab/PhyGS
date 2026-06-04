from glob import glob
import importlib
from inspect import getmodule
import infinigen.assets.objects


if __name__ == '__main__':
    for path in glob('third_party/infinigen/infinigen/assets/objects/*/*.py'):
        category, filename = path.split('/')[-2:]
        filename = filename.split('.')[0]
        if filename == '__init__':
            continue
        factory = filename.title()
        factory = factory.replace('_', '')
        factory = factory + 'Factory'
        try:
            factory = getattr(importlib.import_module(
                f'.{category}.{filename}',
                infinigen.assets.objects.__name__
            ), factory)
        except:
            continue

        print(category, filename, factory.__name__)
        # print(path)
        # print(getmodule(infinigen.assets.objects, ))