

NUMPY_INCLUDE_DIR ?=$(HOME)/.local/lib/python3.10/site-packages/numpy/core/include
EXTRA_FLAGS ?=
OUT_SUFFIX ?= $(shell python3 -c "import distutils; print(distutils.sysconfig.get_config_var('EXT_SUFFIX'))")

PYX_LIST = $(shell find qamreconciliation/ -name "*.pyx" )
OBJ_LIST = $(PYX_LIST:.pyx=$(OUT_SUFFIX))


.PHONY: default clean


default: $(OBJ_LIST)


%$(OUT_SUFFIX): %.pyx $(wildcard *.pxd)
	CFLAGS=-I$(NUMPY_INCLUDE_DIR) cythonize -3 -i $(EXTRA_FLAGS) $<

clean:
	rm $(OBJ_LIST)
