import os
import glob
import shutil

test_suites = glob.glob('../*')
test_suites.remove('../generate_test_data')
test_suites.remove('../shared')

# i gues this removes . and ..
test_suites = [file[3:] for file in test_suites]

print(f'found {test_suites} test setups...')

dont_remove = ['kernels' ]

DELETE_KERNELS = True #Flag to delete kernel files (not folder structure)

for test_case in test_suites:
    files = glob.glob(f'../{test_case}/*')
    
    for file in files:
        files_to_del = [f for f in os.listdir(
            f'{file}') if not (f in dont_remove )]
        print(f'in folder: {file}. want to delete {files_to_del}\n')

        for f in files_to_del:
            path = f'{file}/{f}'
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

        for f in glob.glob(f'{file}/kernels/*'):
            print(f'tring to removek kernel {f}')
            os.remove(f)
