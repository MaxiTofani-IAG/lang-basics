#!/usr/bin/env python3
import pandas as pd
import os
import sys

def split_csv(input_file, chunk_size=100):
    os.makedirs("csv_chunks", exist_ok=True)
    df = pd.read_csv(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        filename = f"csv_chunks/{base_name}_chunk_{i//chunk_size+1:03d}.csv"
        chunk.to_csv(filename, index=False)
        print(f"✓ {filename}: {len(chunk)} filas")

def create_samples(input_file, sizes=[10, 50, 100]):
    os.makedirs("csv_samples", exist_ok=True)
    df = pd.read_csv(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    for size in sizes:
        if size <= len(df):
            sample = df.sample(n=size, random_state=42)
            filename = f"csv_samples/{base_name}_sample_{size}.csv"
            sample.to_csv(filename, index=False)
            print(f"✓ {filename}: {size} filas")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python csv_splitter.py <archivo.csv> [--samples] [--size N]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 100
    
    if "--samples" in sys.argv:
        create_samples(input_file, [10, 50, 100, size])
    else:
        split_csv(input_file, size) 