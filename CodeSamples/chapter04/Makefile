ARCH ?= 80
# $(warning $(GENCODE_FLAG))

ifeq ($(GENCODE_FLAG),)
# Generate SASS code for each SM architecture listed in $(ARCH)
$(foreach sm,$(ARCH),$(eval GENCODE_FLAG += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code for each SM architecture listed in $(ARCH)
$(foreach sm,$(ARCH),$(eval GENCODE_FLAG += -gencode arch=compute_$(sm),code=compute_$(sm)))
endif

NVCC=nvcc
NVCC_FLAGS=$(GENCODE_FLAG)
CU_APPS=globalVariable memTransfer pinMemTransfer readSegment writeSegment simplemathAos2 simplemathSOA2 simpleMathSoA readSegmentUnroll transpose_CopyRow transpose
# CU_APPS=globalVariable memTransfer pinMemTransfer readSegment \
		# readSegmentUnroll simpleMathAoS simpleMathSoA sumArrayZerocpy \
		sumMatrixGPUManaged sumMatrixGPUManual transpose writeSegment
# $(warning $(ARCH))
all: ${CU_APPS}

%: %.cu
	$(NVCC)  $(NVCC_FLAGS) -o $@ $< 

clean:
	rm -f ${CU_APPS}
