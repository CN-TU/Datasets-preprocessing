#!/usr/bin/env python3

# A simple wrapper script for feeding the output of
# multiple subprocess (e.g. unzip) to go-flows

import os
import sys
import subprocess
import shlex

# guess input pcap list as the final part of the parameter string
# with all elements ending with .pcap
last_param = max([ i for i,arg in enumerate(sys.argv) if not arg.endswith('.pcap') ])
goflows = sys.argv[1]
params = sys.argv[1:last_param+1]
filelist = sys.argv[last_param+1:]

unzip_cmd = os.environ['UNZIP_CMD']

pid = os.getpid()

fds = [ os.pipe() for _ in filelist ]

goflows_pid = os.fork()

if goflows_pid == 0:
	for rfd,wfd in fds:
		os.close(rfd)
		os.close(wfd)
	try:
		os.execvp(goflows, params + [ '/proc/%d/fd/%d' % (pid, fd[0]) for fd in fds ])
	except OSError:
		print('Failed to execute file %s' % goflows, file=sys.stderr)
		raise
	
for (rfd, wfd), filename in zip(fds, filelist):
	cmd = '%s %s' % (unzip_cmd, filename.replace(' ', '\\\\ '))
	print (cmd)
	subprocess.check_call(shlex.split(cmd), stdout=wfd)
	os.close(rfd)
	os.close(wfd)
	
_, status = os.waitpid(goflows_pid, 0)

if status != 0:
	sys.exit(1)
