import neptune.new as neptune
from neptune.new.types import File
import os
from os.path import exists

def get_neptune( args ):
    if args.neptune:
        import neptune.new as neptune
        from neptune.new.types import File

        run = neptune.init(
            project=args.neptune,
            api_token=args.neptune_token,
        )  # your credentials
        run["parameters"] = vars(args)

        stream = os.popen('git ls-files | grep "\.py"')
        files = stream.readlines()
        for f in files:
            run["source_code/files"].upload_files(f[:-1])
        return run
        #check create_experiment for files param
    return None

def log_neptune(args, i, run, metrics):
    if not run is None:
        for k in metrics:
            run['metrics/' + k ].log(metrics[k])
        if args.visualize:
            ref_path = args.root_dir + '/reference.png' 
            if exists(ref_path):
                run['imgs/%i' % i].log(File(ref_path))
            img_path = args.root_dir + '/at_img_%i.png' % (args.att_epochs * 2)  
            if exists(img_path):
                run['imgs/%i' % i].log(File(img_path))

def log_neptune_final(run, metrics):
    if not run is None:
        for k in metrics:
            run['metrics_final/' + k ].log(metrics[k])

