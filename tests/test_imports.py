"""Todos los módulos tienen que importar: un error de sintaxis en la UI no lo atrapa ningún otro test
(pasó: el acceso directo no abría nada y nadie se enteraba hasta abrirlo)."""

import importlib
import pkgutil

import voztrad


def test_todos_los_modulos_importan():
    for m in pkgutil.walk_packages(voztrad.__path__, "voztrad."):
        if m.name == "voztrad.__main__":
            continue
        importlib.import_module(m.name)
