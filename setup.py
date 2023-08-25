# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
#
#
# Code Developed by:
# Ahmed A. A. Osman
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
REQUIREMENTS = ["numpy"]

setuptools.setup(
     name='supr',  
     version='0.0.1',
     author="Ahmed A. A. Osman",
     author_email="ahmed.osman@tuebingen.mpg.de",
     install_requires=REQUIREMENTS,
     description="SUPR: Sparse Unified Part-Based Human Representation",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ahmedosman/SUPR",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3",
         "License :: Other/Proprietary License",
         "Operating System :: OS Independent",
     ],
 )