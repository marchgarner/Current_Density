import os.path
import subprocess

folders = [f for f in os.listdir('.') if not os.path.isfile(f)]
path = os.getcwd()

subprocess.call('mkdir spt_data', shell = True, cwd = path)

for n in range(len(folders)):
	if folders[n][-2:] != 'ef' and folders[n] != 'spt_data':
		subprocess.call('mkdir '+ folders[n], shell = True, cwd = path + '/spt_data/')
		subprocess.call('cp *.spt '+ path + '/spt_data/'+folders[n]+'/.', shell = True, cwd = path + '/' + folders[n] +'/data/')
		subprocess.call('cp central_region.xyz '+ path + '/spt_data/'+folders[n]+'/.', shell = True, cwd = path + '/' + folders[n]+'/data/')
		subprocess.call('cp *.png '+ path + '/spt_data/'+folders[n]+'/.', shell = True, cwd = path + '/' + folders[n]+'/data/plots/')
	