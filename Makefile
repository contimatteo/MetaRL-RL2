# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Signifies our desired python version
# Makefile macros (or variables) are defined a little bit differently than traditional bash, keep 
# in mind that in the Makefile there's top-level Makefile-only syntax, and everything else is bash 
# script syntax.
PYTHON = python
PIP = ${PYTHON} -m pip

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "+-----------------------------------------------------------+"
	@echo "|         OS         |  Hardware  |       Setup Command     |"
	@echo "+-----------------------------------------------------------+"
	@echo "|    Apple macOS     |    + M1    |  'make macos.m1.setup'  |"
	@echo "+-----------------------------------------------------------+"

macos.m1.setup:
	pip install -U pip wheel packaging
	zsh ./scripts/mujoco/macos.m1.install.zsh
	pip install -r ./tools/requirements/macos.m1.txt

pip.requirements.clear:
	pip freeze | xargs pip uninstall -y
	pip cache purge

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY: help
.PHONY: macos.m1.setup
.PHONY: pip.requirements.clear
