#!/usr/bin/env python
"""Process a single run with straxen
"""
import argparse
import logging
import time
import os
import sys
import psutil


parser = argparse.ArgumentParser(
    description='Process a single run with straxen',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'run_id', 
    metavar='RUN_ID', 
    type=str,
    help="ID of the run to process; usually the run name.")
parser.add_argument(
    '--context',
    default='strax_workshop_dali',
    help="Name of straxen context to use")
parser.add_argument(
    '--target', 
    default='event_info',
    help='Target final data type to produce')
parser.add_argument(
    '--max_messages', 
    default=2,
    help=("Size of strax's internal mailbox buffers. "
          "Lower to reduce memory usage, at increasing risk of deadlocks."))
parser.add_argument(
    '--workers',
    default=1,
    help=("Number of worker threads/processes. "
          "Strax will multithread (1/plugin) even if you set this to 1."))
parser.add_argument(
    '--multiprocess',
    action='store_true',
    help="Allow multiprocessing.")
parser.add_argument(
    '--shm',
    action='store_true',
    help="Allow passing data via /dev/shm when multiprocessing.")
parser.add_argument(
    '--diagnose_sorting',
    action='store_true',
    help="Diagnose sorting problems during processing")
parser.add_argument(
    '--debug',
    action='store_true',
    help="Enable debug logging to stdout")
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO)

# These imports take a bit longer, so it's nicer
# to do them after argparsing (so --help is fast)
import strax
import straxen

strax.Mailbox.DEFAULT_MAX_MESSAGES = int(args.max_messages)
st = getattr(straxen.contexts, args.context)()
if args.diagnose_sorting:
    st.set_config(dict(diagnose_sorting=True))
st.context_config['allow_multiprocess'] = args.multiprocess
st.context_config['allow_shm'] = args.shm
    
if st.is_stored(args.run_id, args.target):
    print("This data is already available.")
    sys.exit(1)
    
md = st.run_metadata(args.run_id)
t_end = md['end'].timestamp()

process = psutil.Process(os.getpid())

for i, d in enumerate(st.get_iter(
        args.run_id, 
        args.target,
        max_workers=int(args.workers))):
    if i == 0:
        print(f"Received first result: {len(d)} events.", 
              flush=True)
        
        # Compute remaining run duration
        if len(d):
            t_start = d['time'].max() / 1e9
        else:
            t_start = md['start'].timestamp()
        run_duration = t_end - t_start
        
        # Start the clock for ETA measurements
        clock_start = time.time()
        continue

    # Compute detector/data time left
    t = d['time'].max() / 1e9
    dt = t - t_start
    time_left = t_end - t
    
    # Compute processing job clock time left
    clock = time.time()
    d_clock = clock - clock_start
    clock_time_left = time_left / (dt / d_clock)
    
    mem_mb = process.memory_info().rss / 1e6
    
    print(f"Got {len(d)} events. "
          f"Now {dt:.1f} sec / {100 * dt/run_duration:.1f}% into the run. "
          f"ETA {clock_time_left:.2f} sec. Using {mem_mb:.1f} MB RAM.",
          flush=True)
