from setuptools import setup, find_packages

setup(name='diffusionsim',
      version='0.2',
      packages=find_packages(), 
      include_package_data=True,
      package_data={'diffusionsim': ['climsim_data/*']},
      description='Utils for training image diffusion on climsim',
      author='Sammy Agrawal',
      author_email='ssa2206@columbia.edu',
      license='MIT',
      zip_safe=False
)