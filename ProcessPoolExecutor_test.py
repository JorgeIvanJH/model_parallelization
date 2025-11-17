from concurrent.futures import ProcessPoolExecutor, as_completed

def f(x):
    return x * x

if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(f, i) for i in range(4)]
        for fut in as_completed(futures):
            print(fut.result())
