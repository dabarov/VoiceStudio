"""Security-only fixes for the Transformers API families required by engines.

Based on upstream fixes listed in README.md. No runtime monkeypatching and no
model code is imported by the wheel builder. Every edit checks its source shape.
"""
import re


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Unexpected Transformers source around {old[:80]!r}")
    return source.replace(old, new)


def patch(files, version):
    path = "transformers/configuration_utils.py"
    source = files[path].decode()
    anchor = "        # Additional attributes without default values\n"
    source = once(source, anchor, anchor +
        '        kwargs.pop("_attn_implementation_internal", None)\n'
        '        kwargs.pop("_experts_implementation_internal", None)\n')
    files[path] = source.encode()

    # CVE-2026-4372: limit executable attention kernels to the same publisher
    # namespace as the upstream security fix. This path does not exist in 4.52.
    path = "transformers/integrations/hub_kernels.py"
    if version != "4.52.4":
        source = files[path].decode()
        old = '''    return (
        attn_implementation is not None
        and re.search(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", attn_implementation) is not None
    )'''
        new = '''    if attn_implementation is None:
        return False
    match = re.fullmatch(r"(?:paged\\|)?([^/:]+)/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?", attn_implementation)
    return match is not None and match.group(1) == "kernels-community"'''
        files[path] = once(source, old, new).encode()

        # CVE-2026-5241: serialized nested config cannot grant remote-code trust.
        for name in ("configuration_lightglue", "modular_lightglue"):
            path = f"transformers/models/lightglue/{name}.py"
            source = files[path].decode()
            old = '''            if keypoint_detector_config["model_type"] not in CONFIG_MAPPING:
                keypoint_detector_config = AutoConfig.from_pretrained(
                    keypoint_detector_config["_name_or_path"], trust_remote_code=self.trust_remote_code
                )
            else:
                keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](
                    **keypoint_detector_config, attn_implementation="eager"
                )'''
            new = '''            keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](
                **keypoint_detector_config, attn_implementation="eager"
            )'''
            source = once(source, old, new)
            source = once(source, 'self.trust_remote_code = trust_remote_code', 'self.trust_remote_code = False')
            if name == "modular_lightglue":
                source = once(source, 'config.keypoint_detector_config, trust_remote_code=config.trust_remote_code',
                              'config.keypoint_detector_config, trust_remote_code=False')
            files[path] = source.encode()
        path = "transformers/models/lightglue/modeling_lightglue.py"
        files[path] = once(files[path].decode(),
            'config.keypoint_detector_config, trust_remote_code=config.trust_remote_code',
            'config.keypoint_detector_config, trust_remote_code=False').encode()

        # CVE-2026-80047: check existence without downloading, then obtain trust,
        # then copy executable generation code into the cache. 4.52 already does this.
        path = "transformers/generation/utils.py"
        source = files[path].decode()
        start = source.index('        # Fetches the generate.py file from the model repo.')
        end = source.index('        # Handle opt-in `trust_remote_code`', start)
        source = source[:start] + '''        from ..utils import has_file
        if not has_file(pretrained_model_name_or_path, "custom_generate/generate.py", **kwargs):
            raise OSError("Model has no custom generation code")

''' + source[end:]
        source = once(source, '        # Load the custom generate function\n',
            '        module = get_cached_module_file(\n'
            '            pretrained_model_name_or_path, module_file="custom_generate/generate.py", **kwargs\n'
            '        )\n\n        # Load the custom generate function\n')
        files[path] = source.encode()

    # CVE-2026-9856: identical containment check at both template-write sinks.
    for path in ("transformers/processing_utils.py", "transformers/tokenization_utils_base.py"):
        source = files[path].decode()
        pattern = r'(?m)^( +)template_filepath = os.path.join\(chat_template_dir, f"\{template_name\}.jinja"\)\n'
        def guard(match):
            indent = match[1]
            return match[0] + indent + 'if (not isinstance(template_name, str) or "\\\\" in template_name\n' + indent + '        or os.path.dirname(os.path.realpath(template_filepath)) != os.path.realpath(chat_template_dir)):\n' + indent + '    raise ValueError("Invalid chat template name")\n'
        source, count = re.subn(pattern, guard, source)
        if count != 1:
            raise ValueError(f"Unexpected chat template sinks in {path}")
        files[path] = source.encode()

    # CVE-2026-1839: make the restricted loader explicit even on older PyTorch.
    if version.startswith("4."):
        path = "transformers/trainer.py"
        files[path] = once(files[path].decode(), 'checkpoint_rng_state = torch.load(rng_file)',
            'checkpoint_rng_state = torch.load(rng_file, weights_only=True)').encode()

    if version == "4.52.4":
        path = "transformers/models/marian/tokenization_marian.py"
        source = once(files[path].decode(), '''        match = self.language_code_re.match(text)
        code: list = [match.group(0)] if match else []
        return code, self.language_code_re.sub("", text)''', '''        code = []
        if text.startswith(">>") and (end_loc := text.find("<<")) != -1:
            code.append(text[: end_loc + 2])
            text = text[end_loc + 2 :]
        return code, text''')
        files[path] = source.encode()
        path = "transformers/optimization_tf.py"
        source = files[path].decode()
        old = 'if re.search(r, param_name) is not None:'
        if source.count(old) != 2:
            raise ValueError("Unexpected weight decay regex sinks")
        files[path] = source.replace(old, 'if r in param_name:').encode()
        path = "transformers/modeling_tf_pytorch_utils.py"
        source = files[path].decode()
        anchor = '    tf_name = tf_name.replace(":0", "")  # device ids\n'
        files[path] = once(source, anchor, anchor +
            '    if (len(tf_name) > 2048 and "___" in tf_name) or tf_name.count("___") > 10:\n'
            '        raise ValueError("TF variable name is too long or contains too many separators")\n').encode()
        # Importing regex supports possessive quantifiers on Python 3.10 too.
        path = "transformers/models/clvp/number_normalizer.py"
        source = once(files[path].decode(), 'import re\n', 'import regex as re\n')
        for old, new in [
            ('re.compile(r"£([0-9\\,]*[0-9]+)")', 're.compile(r"£([0-9,]*[0-9])")'),
            ('re.compile(r"\\$([0-9\\.\\,]*[0-9]+)")', 're.compile(r"\\$([0-9.,]*[0-9])")'),
            ('re.compile(r"([0-9]+\\.[0-9]+)")', 're.compile(r"([0-9]++\\.[0-9]+)")'),
            ('re.compile(r"[0-9]+(st|nd|rd|th)")', 're.compile(r"[0-9]++(st|nd|rd|th)")'),
        ]:
            source = once(source, old, new)
        files[path] = source.encode()
