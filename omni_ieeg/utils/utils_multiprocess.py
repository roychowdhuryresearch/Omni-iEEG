from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import multiprocessing
import time
import os

def get_folder_size_gb(folder_path):
    """Calculate the total size of a folder in GB using du command"""
    try:
        # Run du command to get size in bytes
        result = subprocess.run(['du', '-sb', folder_path], capture_output=True, text=True, check=True)
        # Extract the size in bytes (first number in the output)
        size_bytes = int(result.stdout.split()[0])
        # Convert to GB
        return size_bytes / (1024**3)
    except (subprocess.SubprocessError, ValueError, IndexError) as e:
        print(f"Error getting folder size: {e}")
        return 0
def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=1, desc=None):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=1): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
            desc (str, default=None): Description for the progress bar
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:], desc=desc)]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        if desc:
            kwargs['desc'] = desc
            
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def worker_wrapper(args):
    """Wrapper function for handling both regular args and kwargs"""
    func, use_kwargs, args_or_kwargs = args
    try:
        if use_kwargs:
            return func(**args_or_kwargs)
        else:
            return func(args_or_kwargs)
    except Exception as e:
        print(f"Error in worker: {e}")
        return e

def robust_parallel_process(array, function, n_jobs=None, use_kwargs=False, front_num=1, desc="Processing"):
    """
    A more robust parallel processing function using multiprocessing.Pool
    
    Args:
        array (array-like): An array to iterate over
        function (function): Function to apply to each element
        n_jobs (int, optional): Number of processes to use. If None, uses cpu_count()
        use_kwargs (bool): Whether to treat array elements as kwargs
        front_num (int): Number of iterations to run serially first
        desc (str): Description for progress bar
        
    Returns:
        list: Results of function applied to each element
    """
    # Use all available CPUs if n_jobs not specified
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    # Run first few items serially to catch any errors early
    front = []
    if front_num > 0 and front_num <= len(array):
        print(f"Running {front_num} items serially first...")
        for i in range(front_num):
            if use_kwargs:
                front.append(function(**array[i]))
            else:
                front.append(function(array[i]))
    
    # If only running serially or very small array, just do everything serially
    if n_jobs == 1 or len(array) <= front_num:
        if len(array) > front_num:
            remaining = array[front_num:]
            results = [function(**a) if use_kwargs else function(a) for a in tqdm(remaining, desc=desc)]
            return front + results
        return front
    
    # Prepare work for parallel processing
    remaining = array[front_num:]
    work_args = [(function, use_kwargs, item) for item in remaining]
    
    # Process in parallel using Pool
    with multiprocessing.Pool(processes=n_jobs) as pool:
        # Use imap_unordered to get results as they complete
        results = list(tqdm(
            pool.imap_unordered(worker_wrapper, work_args),
            total=len(remaining),
            desc=desc
        ))
    
    # Return combined results
    return front + results