#! /bin/sh
echo "Starting O0..."
cd O0
./final_O0
cd ../O0_funroll
./final_O0_funroll

echo "Starting O1..."
cd ../O1
./final_O1
cd ../O1_funroll
./final_O1_funroll

echo "Starting O2..."
cd ../O2
./final_O2
cd ../O2_funroll
./final_O2_funroll

echo "Starting O3..."
cd ../O3
./final_O3
cd ../O3_funroll
./final_O3_funroll