# Makefile for Cilk++ example.  See source code for details.

DIR=$(PWD)
CILKPP	= clang++
LIBARG	= -O2 -g -fopencilk
VIEWLIBARG = -O2 -g -fopencilk -fcilktool=cilkscale
SCLAELIBARG = -O2 -g -fopencilk -fcilktool=cilkscale-benchmark
TARGET	= q1
VIEWTARGET = q1_with_analysis
SCALETARGET = q1_with_analysis_bench

SRC	= $(addsuffix .cpp,$(TARGET))
VIEWSRC	= $(addsuffix .cpp,$(VIEWTARGET))
SCALESRC = $(addsuffix .cpp,$(VIEWTARGET))

all: $(TARGET)

$(TARGET): clean $(SRC)
	$(CILKPP) $(SRC) $(LIBARG) -o $@

$(VIEWTARGET): clean $(VIEWSRC)
	$(CILKPP) $(VIEWSRC) $(VIEWLIBARG) -o $@

$(SCALETARGET): clean $(SCALESRC)
	$(CILKPP) $(SCALESRC) $(VIEWLIBARG) -o $(VIEWTARGET)
	$(CILKPP) $(SCALESRC) $(SCLAELIBARG) -o $@

clean:
	rm -f $(TARGET) $(VIEWTARGET) $(SCALETARGET)

test: $(TARGET)
	CILK_NWORKERS=6  ./q1

view: $(VIEWTARGET)
	CILK_NWORKERS=6  ./q1_with_analysis

scale: $(SCALETARGET)
	(python3 $(OpenCilk_HOME)/share/Cilkscale_vis/cilkscale.py --cilkscale   $(DIR)/$(VIEWTARGET) --cilkscale-benchmark $(DIR)/$(SCALETARGET) --args 1024 16 -oplot 0,1 )