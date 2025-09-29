# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations for DoubleBee robot."""

import os

# Conveniences to other module directories via relative paths
DOUBLEBEE_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
"""Path to the extension source directory."""

DOUBLEBEE_ASSETS_DATA_DIR = os.path.join(DOUBLEBEE_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

# Configure the module-level variables
__version__ = "0.1.0"


##
# Configuration for different assets.
##

from .doublebee_v1 import *
