"""CPU-only installed Mach-O identity contracts; no actual loaders or inference."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import bench_lookup_staircut as r


def load_commands(dependencies=(), rpaths=(), install_id=None):
    blocks = []
    for kind, label, values in (("LC_ID_DYLIB", "name", [install_id] if install_id else []),
                                ("LC_RPATH", "path", rpaths), ("LC_LOAD_DYLIB", "name", dependencies)):
        for value in values:
            blocks.append(f'Load command {len(blocks)}\n      cmd {kind}\n      {label} {value} (offset 24)\n')
    listing = ''.join('\t' + value + ' (compatibility version 1.0.0, current version 1.0.0)\n'
                      for value in ([install_id] if install_id else []) + list(dependencies))
    return listing, ''.join(blocks)


class RuntimeIdentity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.exe = self.root / 'bin/server'
        self.mlx = self.root / 'lib/libmlx.dylib'
        self.child = self.root / 'lib/sub/libhelper.dylib'
        self.metal = self.root / 'lib/mlx.metallib'
        for path in (self.exe, self.mlx, self.child, self.metal):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original ' + path.name.encode())
        self.metadata = {
            str(self.exe): load_commands(('@rpath/libmlx.dylib', '/usr/lib/libSystem.B.dylib'), ('@loader_path/../lib',)),
            str(self.mlx): load_commands(('@rpath/libhelper.dylib',), ('@loader_path/sub',), '@rpath/libmlx.dylib'),
            str(self.child): load_commands(('/System/Library/Frameworks/Metal.framework/Metal',), (), '@rpath/libhelper.dylib')}
        self.calls = []
        def run(command, **kwargs):
            self.calls.append(command)
            if command == ['/usr/bin/sw_vers']:
                return SimpleNamespace(stdout='ProductVersion: 27.0\nBuildVersion: fixture\n')
            listing, commands = self.metadata[command[-1]]
            return SimpleNamespace(stdout=listing if '-L' in command else commands)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(r.sys, 'platform', 'darwin').start()
        mock.patch.object(r.subprocess, 'run', side_effect=run).start()

    def test_recurses_loader_rpaths_and_records_system_provenance(self):
        identity = r.runtime_identity(self.exe)
        self.assertEqual(set(identity['images']), {str(self.exe), str(self.mlx), str(self.child)})
        self.assertEqual(identity['images'][str(self.child)]['effective_rpaths'],
                         [str(self.root / 'lib/sub'), str(self.root / 'lib')])
        self.assertIn('/usr/lib/libSystem.B.dylib', identity['system_libraries'])
        self.assertIn('BuildVersion: fixture', identity['os_version'])
        self.assertTrue(identity['metal_assets'][str(self.metal)]['exists'])
        self.assertTrue(all(command[0] in ('/usr/bin/otool', '/usr/bin/sw_vers') for command in self.calls))

    def test_network_hostname_changes_do_not_invalidate_runtime_but_kernel_changes_do(self):
        original = os.uname_result(('Darwin', 'before.local', '27.0.0', 'kernel-build-a', 'arm64'))
        with mock.patch.object(r.os, 'uname', return_value=original):
            identity = r.runtime_identity(self.exe)
        changed_host = os.uname_result(('Darwin', 'after.local', '27.0.0', 'kernel-build-a', 'arm64'))
        with mock.patch.object(r.os, 'uname', return_value=changed_host):
            self.assertEqual(identity, r.runtime_identity(self.exe))
        for index, replacement in ((0, 'OtherOS'), (2, '27.1.0'), (3, 'kernel-build-b'), (4, 'x86_64')):
            changed = list(original)
            changed[index] = replacement
            with self.subTest(field=index), mock.patch.object(r.os, 'uname', return_value=os.uname_result(changed)):
                self.assertNotEqual(identity, r.runtime_identity(self.exe))

    def test_transitive_library_and_metallib_content_changes_invalidate(self):
        initial = r.runtime_identity(self.exe)
        self.child.write_bytes(b'replaced transitive kernel code')
        self.assertNotEqual(r.runtime_identity(self.exe), initial)
        initial = r.runtime_identity(self.exe)
        self.metal.write_bytes(b'replaced Metal code')
        self.assertNotEqual(r.runtime_identity(self.exe), initial)

    def test_embedded_absolute_metallib_and_new_candidate_are_frozen(self):
        embedded = self.root / 'compiled/mlx.metallib'
        embedded.parent.mkdir()
        embedded.write_bytes(b'embedded fallback')
        self.mlx.write_bytes(b'header\x00' + str(embedded).encode() + b'\x00')
        initial = r.runtime_identity(self.exe)
        self.assertTrue(initial['metal_assets'][str(embedded)]['exists'])
        embedded.write_bytes(b'changed fallback')
        self.assertNotEqual(r.runtime_identity(self.exe), initial)
        initial = r.runtime_identity(self.exe)
        (self.exe.parent / 'default.metallib').write_bytes(b'new overriding candidate')
        self.assertNotEqual(r.runtime_identity(self.exe), initial)

    def test_unresolved_transitive_dependency_fails_closed(self):
        self.child.unlink()
        with self.assertRaisesRegex(r.BenchError, 'unresolved non-system dependency'):
            r.runtime_identity(self.exe)

    def test_missing_all_metal_assets_fails_closed(self):
        self.metal.unlink()
        with self.assertRaisesRegex(r.BenchError, 'Metal-library asset'):
            r.runtime_identity(self.exe)

    def test_symlink_target_and_loading_configuration_are_part_of_identity(self):
        alternate = self.root / 'lib/sub/alternate.dylib'
        alternate.write_bytes(self.child.read_bytes())
        initial = r.runtime_identity(self.exe)
        self.child.unlink()
        self.child.symlink_to(alternate)
        changed = r.runtime_identity(self.exe)
        self.assertNotEqual(initial, changed)
        self.assertIn(str(alternate), changed['images'])

    def test_environment_cannot_override_frozen_linkage(self):
        with mock.patch.dict(os.environ, {'DYLD_LIBRARY_PATH': '/other', 'DYLD_INSERT_LIBRARIES': '/inject',
                                         'LD_LIBRARY_PATH': '/other', 'LD_PRELOAD': '/inject',
                                         'MLX_METAL_PATH': '/other', 'KEEP_ME': 'yes'}):
            environment = r.clean_environment('legacy')
        self.assertFalse(any(key.startswith('DYLD_') for key in environment))
        self.assertNotIn('LD_LIBRARY_PATH', environment)
        self.assertNotIn('LD_PRELOAD', environment)
        self.assertNotIn('MLX_METAL_PATH', environment)
        self.assertEqual(environment['KEEP_ME'], 'yes')


if __name__ == '__main__':
    unittest.main()
