#!/bin/bash
set -e

time nice python run_sim.py
echo 'rendering'
python plot_res.py
