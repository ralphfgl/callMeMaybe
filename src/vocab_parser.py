from typing import Any
import json
import numpy as np
import string
import re


def get_allowed_chars(current_str: str, allowed_names: list[str]) -> list[str]:
    """Determine the next valid characters based on the JSON string state.

    Args:
        current_str (str): The partially generated JSON string.
        allowed_names (list[str]): List of valid function names.
    Returns:
        list[str]: A list of allowed character sequences for the next step.
    """

    # Phase 1
    prefix = '{"name":"'
    # returns the characters of the prefix sequence that havent been generated yet
    if len(current_str) < len(prefix):
        return [prefix[len(current_str) :]]

    # Phase 2: The function name
    # after prefix is chars generated after the prefix
    after_prefix = current_str[len(prefix) :]
    # if function name is not completed
    # complete the generation based on function allowed names
    # if there is multiple match, returns all valid continuation, the model will decide which one it will prefer
    # and we restrict the model vocab with token that start with one of the string
    if '"' not in after_prefix:
        return [
            name[len(after_prefix) :] + '"'
            for name in allowed_names
            if name.startswith(after_prefix)
        ]

    # Phase 3: The bridge
    # function name has been completed
    # it split after last ", so [0] is the func name and [1] is empty
    func_name = after_prefix.split('"')[0]
    # add bridge
    target = prefix + func_name + '","parameters":{'
    if len(current_str) < len(target):
        return [target[len(current_str) :]]

    # Phase 4: The arguments
    # allow only printable chars
    return list(string.printable)


def generate_constrained_json(prompt_text: str, cache: Any) -> str:
    """Generate a valid function call JSON using constrained decoding.

    Args:
        prompt_text (str): The natural language user prompt.
        cache (Any): MaskCache instance with model and pre-computed masks.

    Returns:
        str: The generated valid JSON string representing the function call.
    """

    # we put function name, description and parameter in optimize schema
    optimized_schemas = []
    for f in cache.raw_functions:
        optimized_schemas.append(
            {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {}),
            }
        )

    # Remove ALL spaces from JSON string, so it has less token
    schema_hints = json.dumps(optimized_schemas, separators=(",", ":"))

    prompt = (
        f"System: You are a strict API. Output ONLY valid JSON matching "
        f"these schemas: {schema_hints}\n"
        r"Rule: For the regex field, NEVER output literal matches. "
        r"Always use proper regex sets "
        r"(e.g. '[aeiouAEIOU]', '[0-9]+', '\\bword\\b'). "
        r"For replacement, if asked for a character (e.g. asterisks), "
        r"output EXACTLY ONE character (e.g. '*')."
        "\n"
        f"User: {prompt_text}\n"
        "Tool Call: "
    )

    # tokenize the prompt
    input_ids = cache.model.encode(prompt).tolist()[0]
    # get vocab size (all possible next token) to construct a mask
    vocab_size = len(cache.model.get_logits_from_input_ids(input_ids))

    # no generated output in the beginning
    current_str = ""

    # manually insert beginning
    prefix = '{"name":"'
    current_str = prefix

    # extend() -> builtin list method, add each element of another iterable (while append() whould just append a list as one element of the list [1, 2, 3, [4, 5]] )
    input_ids.extend(cache.model.encode(prefix).tolist()[0])
    bridge_injected = False

    # prevent infinite generation
    max_tokens = 150

    # it continue until either : json if finished (end with }} or max generation length is reached)
    while (
        not current_str.replace(" ", "").replace("\n", "").endswith("}}")
        and len(input_ids) < len(prompt) + max_tokens
    ):
        # prefix completion using the known set of valid function name
        if prefix in current_str and '","parameters":{' not in current_str:
            after_prefix = current_str.split(prefix)[1]
            possible_names = [
                n for n in cache.allowed_fn if n.startswith(after_prefix)
            ]

            # second condition ensure not to do something unnecessary if the complete name has already been geenrated
            if len(possible_names) == 1 and possible_names[0] != after_prefix:
                remainder = possible_names[0][len(after_prefix) :] + '"'
                current_str += remainder
                input_ids.extend(cache.model.encode(remainder).tolist()[0])
                continue

        # Bridge Fast-Forward
        if (
            current_str.endswith('"')
            and not bridge_injected
            and prefix in current_str
            and len(current_str) > len(prefix)
        ):
            bridge = ',"parameters":{'
            current_str += bridge
            input_ids.extend(cache.model.encode(bridge).tolist()[0])
            bridge_injected = True

            func_name = current_str.split('"name":"')[1].split('"')[0]
            # 67 is a sentinel value, is returned if func_name is not found
            # finish the json if the function has no params
            if cache.func_params.get(func_name, 67) == 0:
                current_str += "}}"
                break

            # generator expression with next, searching thru  a list
            # return None if no match found (instead of rasingn error)
            active_schema = next(
                (f for f in cache.raw_functions if f["name"] == func_name),
                None,
            )
            # NOTE: clearer but more verbose
            # active_schema = None
            # for f in cache.raw_functions:
            #     if f["name"] == func_name:
            #         active_schema = f
            #         break

            # switch from all the function to one active schema and update the prompt
            # we completely refocus on the function params
            if active_schema:
                tiny_schema = json.dumps(
                    [
                        {
                            "name": active_schema["name"],
                            "description": active_schema.get(
                                "description", ""
                            ),
                            "parameters": active_schema.get("parameters", {}),
                        }
                    ],
                    separators=(",", ":"),
                )

                tiny_prompt = (
                    f"System: Output valid JSON matching this schema: "
                    f"{tiny_schema}\n"
                    r"Rule: For the regex field, NEVER output "
                    r"literal matches. "
                    r"Always use proper regex sets "
                    r'(e.g. "[aeiouAEIOU]", "[0-9]+", "\\bword\\b"). '
                    r"For replacement, if asked for a character "
                    r"(e.g. asterisks), output EXACTLY ONE character "
                    r"(e.g. '*')."
                    "\n"
                    f"User: {prompt_text}\n"
                    f"Tool Call: {current_str}"
                )

                input_ids = cache.model.encode(tiny_prompt).tolist()[0]
            else:
                input_ids.extend(cache.model.encode(bridge).tolist()[0])

            continue

        # determines what token are allowed next (and return a list)
        rules = get_allowed_chars(current_str, cache.allowed_fn)
        logits = np.array(cache.model.get_logits_from_input_ids(input_ids))
        mask = np.zeros(vocab_size, dtype=bool)

        # if more than 10 allowed char, enter the quota and type shield
        if len(rules) > 10:
            # --- PHASE 4: THE QUOTA & TYPE SHIELD ---

            # extract the current function name
            # name -> whitespace -> : -> whitespace -> capture everything until next "
            # .search(pattern, string)
            match = re.search(r'"name"\s*:\s*"([^"]+)', current_str)
            # group is used with regex match objects
            # .group(n) -> return nth group captured
            func_name = match.group(1) if match else ""

            # split at parameter and take what after
            params_str = (
                current_str.split('"parameters"')[1]
                if '"parameters"' in current_str
                else ""
            )

            # parse parameter char by char
            # we want to track if the model is prompting a value or has finished and is waiting for a key
            if params_str:
                in_string = False
                last_structural_colon = -1
                last_structural_comma = -1
                last_structural_brace = -1

                # track position of structural chars, while ignoring those inside string literal
                # in string logic toggle when an unescaped " is found
                for i, char in enumerate(params_str):
                    if char == '"':
                        if i == 0 or params_str[i - 1] != "\\":
                            in_string = not in_string
                    elif not in_string:
                        if char == ":":
                            last_structural_colon = i
                        elif char == ",":
                            last_structural_comma = i
                        elif char == "}":
                            last_structural_brace = i

                # determine if we are in a value
                is_inside_value = (
                    last_structural_colon > last_structural_comma
                    and last_structural_colon > last_structural_brace
                )

                active_key = ""
                if is_inside_value:
                    keys_found = re.findall(r'"([^"]+)"\s*:', params_str)
                    if keys_found:
                        active_key = keys_found[-1]

                expected_type = cache.param_types.get(func_name, {}).get(
                    active_key, "Any"
                )

                param_count = len(re.findall(r'"([^"]+)"\s*:', params_str))
                target_count = cache.func_params.get(func_name, 99)

                # --- THE MASK ROUTER ---

                if is_inside_value and expected_type == "number":
                    mask = cache.p4_numbers_only.copy()

                    if param_count == target_count:
                        for i, s in cache.clean_dict_items:
                            if "," in s:
                                mask[i] = False

                # String value state
                elif is_inside_value and in_string:
                    mask = cache.p4_mask.copy()

                    if param_count == target_count:
                        for i, s in cache.clean_dict_items:
                            if '",' in s.replace(" ", ""):
                                mask[i] = False

                    if active_key == "regex":
                        for i, s in cache.clean_dict_items:
                            if " " in s:
                                mask[i] = False

                        if re.search(r'"regex"\s*:\s*"$', params_str):
                            for i, s in cache.clean_dict_items:
                                if not any(
                                    s.startswith(c) for c in ["[", "\\"]
                                ):
                                    mask[i] = False

                elif param_count == target_count:
                    clean_str = current_str.strip()
                    if clean_str.endswith('"'):
                        current_str = clean_str + "}}"
                        print(
                            f"\rGenerating: {current_str}", end="", flush=True
                        )
                        break
                    elif clean_str.endswith("}"):
                        current_str = clean_str + "}"
                        print(
                            f"\rGenerating: {current_str}", end="", flush=True
                        )
                        break
                    elif clean_str.endswith(","):
                        current_str = clean_str[:-1] + "}}"
                        print(
                            f"\rGenerating: {current_str}", end="", flush=True
                        )
                        break
                    else:
                        mask = cache.p4_no_comma.copy()

                else:
                    mask = cache.p4_mask.copy()
                    is_expecting_key = params_str.strip().endswith(
                        "{"
                    ) or params_str.strip().endswith(",")
                    if is_expecting_key:
                        for i, s in cache.clean_dict_items:
                            cleaned = s.strip()
                            if not (cleaned.startswith('"') or not cleaned):
                                mask[i] = False

        # Phase 1-3: Strict Spelling
        else:
            for i, s in cache.mini_dict:
                if any(rule.startswith(s) for rule in rules):
                    mask[i] = True

        logits[~mask] = -np.inf

        best_id = int(np.argmax(logits))
        current_str += cache.vocab_dict.get(best_id, "")
        input_ids.append(best_id)

        if (
            current_str.endswith('"')
            and not bridge_injected
            and prefix in current_str
        ):
            bridge = ',"parameters":{'
            current_str += bridge
            input_ids.extend(cache.model.encode(bridge).tolist()[0])
            bridge_injected = True
            continue

        else:
            print(f"\rGenerating: {current_str}", end="", flush=True)

    print()
    return current_str
