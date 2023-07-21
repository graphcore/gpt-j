# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
""" Copies over resources from one repository to another

```yaml
notebook-tutorials-text_classification:
    source:
        # This repository will be cloned if it is an URL or if it is a folder
        # it is expected to be a git repository where the branch can be
        # changed
        repository: github.com/graphcore/tutorials
        Optional
        ref: branch or commit
        # The prefix is stripped from all the file names
        prefix: tutorials/pytorch/basics
        # List of paths that will be copied
        paths:
            # Can either be the path to a file that needs to be copied
            - tutorials/pytorch/basics/walkthrough.ipynb
            # or an entry which lets you use glob expressions for capturing
            # paths, it is interpreted in python as Path(path).rglob(expression)
            - recursive: true
              expression: *
              path: tutorials/pytorch/basics/

    target:
        repository: github.com/gradient-ai/Graphcore-Pytorch
        prefix: tutorial-notebooks/basics/
        rename: []
```

"""
from typing import NamedTuple, List, Union, Optional, Dict, Any, Tuple, Set
from pathlib import Path
import git
import re
import yaml
import datetime
import time
import shutil
import json
import warnings


DictOrList = Union[Dict, List, Tuple]

TMP_FOLDER = Path(".").resolve() / "clones"
TMP_FOLDER.mkdir(parents=True, exist_ok=True)


class DictListGetter:
    """Access a list or a dictionary as if it was a dict

    This is used to parse the argument lists in the `from_dict` methods
    of NamedTuples used to parse the configuration from YAML to the pattern
    of classes used in the deployment pipeline.
    """

    def __init__(
        self,
        dict_or_list: DictOrList,
        field_order: Union[List[str], Tuple[str, ...]],
        run_checks=True,
    ) -> None:
        self._dict_or_list = dict_or_list
        self._field_order = {f: i for i, f in enumerate(field_order)}
        if run_checks:
            self.check()

    def check(self):
        if isinstance(self._dict_or_list, list) and len(self._dict_or_list) > len(self._field_order):
            raise ValueError("More list arguments than their are fields")
        elif isinstance(self._dict_or_list, dict):
            extra_keys = [k for k in self._dict_or_list if k not in self._field_order]
            if extra_keys:
                raise ValueError(f"Additional keys {extra_keys} were provided")

    def __getitem__(self, __key):
        if isinstance(self._dict_or_list, list):
            return self._dict_or_list[self._field_order[__key]]
        return self._dict_or_list[__key]

    def __contains__(self, __key):
        try:
            self[__key]
            return True
        except (KeyError, IndexError):
            return False

    def get(self, __key, default=None):
        try:
            return self[__key]
        except (KeyError, IndexError):
            return default


def namedtuple_dict_encoder(o: Any):
    """Processes named tuples as dictionaries in nested object structures"""
    if hasattr(o, "_asdict"):
        return namedtuple_dict_encoder(o._asdict())
    if isinstance(o, dict):
        return {k: namedtuple_dict_encoder(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(namedtuple_dict_encoder(v) for v in o)
    return o


class JsonEncoder(json.JSONEncoder):
    """JSON serialiser which supports pathlib.Path"""

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


# ================================================
#  The classes in this section encode the schema of
# YAML in a set of python classes.


class PathsToCopy(NamedTuple):
    path: Path
    expression: Optional[str] = "*"
    recursive: Optional[bool] = True

    @classmethod
    def from_dict(cls, d_in: DictOrList):
        d = DictListGetter(d_in, cls._fields)
        arg_list = {f: d[f] for f in cls._fields if f in d}
        arg_list["path"] = Path(d["path"])
        return cls(**arg_list)

    def get_paths(self, root_dir: Path) -> List[Path]:
        path = root_dir / self.path
        if not path.exists():
            raise ValueError(f"{path} does not exists, {self} is invalid")
        if path.is_dir():
            if self.expression is None or self.recursive is None:
                raise ValueError("Folders need 'expression' and 'recursive' to be set")
            iterator = path.rglob if self.recursive else path.glob
            return [*iterator(self.expression)]
        return [path]


class Repository(NamedTuple):
    origin: str
    ref: Optional[str]
    prefix: Path

    @classmethod
    def from_dict(cls, d_in: DictOrList):
        d = DictListGetter(d_in, cls._fields)
        return cls(
            origin=d["origin"],
            ref=d["ref"],
            prefix=Path(d["prefix"]),
        )

    def prepare(self, cloning_directory: Path = TMP_FOLDER) -> Path:
        """Clones and checkouts the correct ref of the origin"""
        # Treat the origin as a folder, if it doesn't exist it's a URL to clone
        repo_folder = Path(self.origin)
        if not repo_folder.exists():
            repo_folder = cloning_directory / self._sanitised_url()

        # Clone the repository or make sure that the folder is a repo.
        if not repo_folder.exists():
            repo = git.Repo.clone_from(self.origin, to_path=repo_folder)
        else:
            try:
                repo = git.Repo(repo_folder)
            except git.InvalidGitRepositoryError as error:
                raise git.InvalidGitRepositoryError(
                    f"{repo_folder} is not a git repository. If this folder"
                    "was cloned make sure the clone was successful, or if it is meant to be"
                    "a local repository make sure to run `git init` in the folder before "
                    "calling `prepare` on that path."
                ) from error
        # if a ref is specified, try to fetch it then try to check it out
        if repo.remotes and self.ref:
            try:
                repo.git.fetch()
            except git.GitCommandError as error:
                warnings.warn(
                    f"Failed to fetch the repository {self.origin} in folder"
                    f" {repo_folder}. Trying to fetch raised: {error}"
                )
        if self.ref:
            repo.git.checkout(self.ref)
            if not repo.head.is_detached:
                repo.git.pull()

        return repo_folder

    def _sanitised_url(self) -> str:
        return "".join([c if re.match("[a-zA-Z0-9]", c) else "-" for c in str(self.origin)])


class Source(NamedTuple):
    repository: Repository
    paths: List[PathsToCopy]
    excludes: List[PathsToCopy] = []

    @classmethod
    def from_dict(cls, d_in: DictOrList):
        d = DictListGetter(d_in, cls._fields)
        optional = dict(excludes=[PathsToCopy.from_dict(p) for p in d["excludes"]]) if "excludes" in d else {}
        return cls(
            repository=Repository.from_dict(d["repository"]),
            paths=[PathsToCopy.from_dict(p) for p in d["paths"]],
            **optional,
        )

    def get_source_paths(self, search_path: Path) -> Dict[str, Path]:
        file_paths: List[Path] = []
        for path_entry in self.paths:
            file_paths.extend(path_entry.get_paths(search_path))
        exclude_paths: Set[Path] = set()
        for path_exclude in self.excludes:
            exclude_paths = exclude_paths.union(path_exclude.get_paths(search_path))
        file_paths = [p for p in file_paths if p not in exclude_paths]
        # dictionary is file-names relative to prefix
        # mapping to exact source path
        source_entries = {}
        for file in file_paths:
            if file.is_dir():
                continue
            name = file.relative_to(search_path)
            try:
                name = name.relative_to(self.repository.prefix)
            except ValueError:
                name = name.name
            name = str(name)
            if name in source_entries and source_entries[name] != file:
                raise RuntimeError(f"Entry {name} for file '{source_entries[name]}' would be overwritten with {file}")
            source_entries[name] = file

        return source_entries


class PathCopy(NamedTuple):
    source: Path
    target: Path

    @classmethod
    def from_dict(cls, d_in: DictOrList):
        d = DictListGetter(d_in, cls._fields)

        return cls(
            source=Path(d["source"]),
            target=Path(d["target"]),
        )


class Target(NamedTuple):
    repository: Repository
    renames: Dict[str, Path]

    @classmethod
    def from_dict(cls, d_in: DictOrList):
        d = DictListGetter(d_in, cls._fields)
        return cls(
            Repository.from_dict(d["repository"]),
            {k: Path(v) for k, v in d.get("renames", {}).items()},
        )

    def define_target_paths(self, target_directory: Path, source_entries: Dict[str, Path]) -> Dict[str, PathCopy]:
        return {
            k: PathCopy(
                source=v,
                target=target_directory
                / self.renames.get(
                    f"{self.repository.prefix}/{k}",
                    f"{self.repository.prefix}/{k}",
                ),
            )
            for k, v in source_entries.items()
        }


class CopyConfig(NamedTuple):
    source: Source
    target: Target

    @classmethod
    def from_dict(cls, d_in: DictOrList):
        d = DictListGetter(d_in, cls._fields)

        return cls(
            source=Source.from_dict(d["source"]),
            target=Target.from_dict(d["target"]),
        )

    def prepare(self):
        source_path = self.source.repository.prepare()
        target_path = self.target.repository.prepare()
        to_copy = self.source.get_source_paths(source_path)
        to_copy = self.target.define_target_paths(target_path, to_copy)
        return to_copy


# ======================================================
# Uses the schema to parse and process the files to copy


def copy(to_copy: Dict[str, PathCopy], revert: bool = False, dry_run: bool = False):
    def dry_run_copy(*args, **kwargs):
        print(f"[dry run] copyfile({args}, {kwargs})")
        return 0

    if dry_run:
        copy_fn = dry_run_copy
    else:
        copy_fn = shutil.copyfile

    for key, entry in to_copy.items():
        print(f"copying entry: {key}")
        assert entry.source.exists(), f"Source didn't exist for {key}: {entry}"
        entry.target.parent.mkdir(parents=True, exist_ok=True)
        if revert:
            copy_fn(entry.target, entry.source, follow_symlinks=True)
        else:
            copy_fn(entry.source, entry.target, follow_symlinks=True)


def parse_yaml_configs(yaml_files) -> Dict[str, CopyConfig]:
    spec = {}
    spec_sources = {}
    for yaml_file in yaml_files:
        print(f"Examining: '{yaml_file}'")
        found_configs = yaml.load(open(yaml_file).read(), Loader=yaml.Loader)
        # strip out configs which start with underscore
        found_configs = {k: v for k, v in found_configs.items() if k[0] != "_"}
        # Add the file each benchmark config came from
        for v in found_configs:
            spec_sources[v] = yaml_file
        spec.update(found_configs)

    configs: Dict[str, CopyConfig] = {}
    for name, dict_config in spec.items():
        configs[name] = CopyConfig.from_dict(dict_config)
    return configs


def main(named_configs: Dict[str, CopyConfig], revert: bool, dry_run: bool = False):
    to_copy = {}
    for name, conf in named_configs.items():
        print(f"Preparing config: {name}")
        to_copy[name] = conf.prepare()

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H.%M.%S.%f")
    file_name = f"deployment_log_{timestamp}.json"
    json.dump(
        namedtuple_dict_encoder({"config": named_configs, "copies": to_copy}),
        open(file_name, "w"),
        cls=JsonEncoder,
        indent=2,
    )
    print(f"Configuration and report logged to: {file_name}")

    for name, pairs in to_copy.items():
        print(f"Copying config: {name}")
        copy(pairs, revert)
    return to_copy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        default=[str(TMP_FOLDER / "test-configs.yaml")],
        type=str,
        nargs="+",
        help="Yaml files with copying spec",
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Copy files updated in destination repository to origin repository",
    )
    parser.add_argument("--dry-run", action="store_true", help="Does everything except copying the files")
    args = parser.parse_args()

    configs = parse_yaml_configs(args.spec)
    main(configs, args.revert, args.dry_run)