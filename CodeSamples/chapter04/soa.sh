#!/bin/bash

echo this is result of simplemath.txt > soa.txt

nsight_cmp --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct simplemathAos2 >> soa.txt
nsight_cmp --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct simplemathSOA2 >> soa.txt

nsight_cmp --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct simpleMathSoA >> soa.txt
