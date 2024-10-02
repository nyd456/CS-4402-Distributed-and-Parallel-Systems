NVCC = nvcc
NVCCFLAGS = -O2

EXE = a2

SRC = a2.cu

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(EXE)

run:
	./$(EXE)

run1:
	./$(EXE) 1

run2:
	./$(EXE) 2