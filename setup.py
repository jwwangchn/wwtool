import os
import platform
import subprocess
import time
import numpy as np
from setuptools import find_packages, setup, Extension

from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


MAJOR = 0
MINOR = 0
PATCH = 1
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

version_file = 'wwtool/version.py'


def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from wwtool.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}

__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }
    print(name, module, sources)
    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension

if __name__ == '__main__':
    write_version_py()
    setup(
        name='wwtool',
        version=get_version(),
        description='Tools for jwwangchn Research',
        # long_description=readme(),
        keywords='computer vision, object detection',
        url='https://github.com/jwwangchn/wwtool',
        packages=find_packages(),
        package_data={'wwtool.csrc': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        setup_requires=['cython', 'numpy'],
        # tests_require=['pytest'],
        install_requires=[
            'mmcv>=0.2.6', 'numpy', 'matplotlib', 'six', 'terminaltables',
            'pycocotools', 'pySerial', 'shapely', 'geojson', 'scikit-image==0.14.2', 'geopandas', 'rasterio'
        ],
        ext_modules=[
            make_cython_ext(
                name='bbox_overlaps_cpu',
                module='wwtool.csrc.bbox',
                sources=['src/bbox_overlaps_cpu.pyx']),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)