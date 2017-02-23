from numpy.distutils.core import setup

from numpy.distutils.core import Extension

ext = Extension(name='fext',
                sources=['fext.f95'],
                extra_link_args=['-lgomp']
               )
setup(name='fext', ext_modules=[ext])
