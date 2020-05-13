import glob

paths = glob.glob("./data/tiny-imagenet-200/val/**/*.JPEG")
with open('test_eval.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
	for i in range(len(paths)):
		eval_output_file.write('{},{},{},{},{}\n'.format(i, paths[i],64,64,3))
