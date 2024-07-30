

NUMPY_INCLUDE_DIR ?=$(HOME)/.local/lib/python3.10/site-packages/numpy/core/include
EXTRA_FLAGS ?=
OUT_SUFFIX ?= $(shell python3 -c "import distutils; print(distutils.sysconfig.get_config_var('EXT_SUFFIX'))")

PYX_LIST = $(shell find qamreconciliation/ -name "*.pyx" -not -path "**/.ipynb_checkpoints/**" )
OBJ_LIST = $(PYX_LIST:.pyx=$(OUT_SUFFIX))
C_LIST = $(PYX_LIST:.pyx=.c)

RECIPE = CFLAGS=-I$(NUMPY_INCLUDE_DIR) cythonize -3 -i $(EXTRA_FLAGS) $<


.PHONY: default clean


default: $(OBJ_LIST)


%$(OUT_SUFFIX): %.pyx %.pxd
	$(RECIPE)

%$(OUT_SUFFIX): %.pyx
	$(RECIPE)


clean:
	rm -rf $(OBJ_LIST) $(C_LIST)
