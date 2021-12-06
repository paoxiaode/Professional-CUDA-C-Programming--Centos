ARCH ?= 80

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(ARCH)
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code for each SM architecture listed in $(ARCH)
$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=compute_$(sm)))
endif

NVCC=nvcc
NVCC_FLAGS=$(GENCODE_FLAGS)
CU_APPS=nestedHelloWorld 
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	$(NVCC)  $(NVCC_FLAGS) $< -o $@ -lcudadevrt --relocatable-device-code true
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
