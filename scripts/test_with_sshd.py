"""Run the complete backend suite with an isolated localhost SSH server.

Uses temporary keys/config and a random loopback port; never changes ~/.ssh or
system SSH settings. Requires an installed sshd and permission to bind localhost.
"""
import getpass
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def main():
    binary = shutil.which('sshd') or '/usr/sbin/sshd'
    with tempfile.TemporaryDirectory(prefix='agx-ssh-', dir='/tmp') as folder:
        root = Path(folder)
        for name in ['host', 'client']:
            subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(root/name)], check=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port=sock.getsockname()[1]
        config = root/'sshd_config'
        config.write_text(f'''ListenAddress 127.0.0.1
Port {port}
HostKey {root/'host'}
PidFile {root/'pid'}
AuthorizedKeysFile {root/'client.pub'}
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
StrictModes no
AllowUsers {getpass.getuser()}
LogLevel VERBOSE
''')
        with (root/'server.log').open('w') as log:
            server = subprocess.Popen([binary, '-D', '-e', '-f', str(config)], stdout=log, stderr=log)
            try:
                for _ in range(100):
                    if server.poll() is not None:
                        raise RuntimeError((root/'server.log').read_text())
                    try:
                        with socket.create_connection(('127.0.0.1', port), timeout=.2): break
                    except OSError: time.sleep(.05)
                env = {**os.environ, 'AGENTICA_TEST_SSH_PORT':str(port),
                       'AGENTICA_TEST_SSH_KEY':str(root/'client'), 'AGENTICA_TEST_SSH_KNOWN_HOSTS':str(root/'known_hosts')}
                result = subprocess.call([sys.executable, '-m', 'pytest', 'tests', '-q', '-rs'], env=env)
                if result: print((root/'server.log').read_text()[-4000:])
                return result
            finally:
                server.terminate(); server.wait(timeout=5)

if __name__ == '__main__': sys.exit(main())
