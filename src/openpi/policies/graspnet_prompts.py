"""Paraphrase, category-generic and misspelling variants for the pick-up prompts.

Used by pi05_Franka_GRASPNET_FINAL_moreData via transforms.PromptVariants, which swaps
each sample's canonical task string for one of these during training. The goal is a
policy that generalizes over phrasing and over object *category* instead of memorizing
thirteen exact strings, and that does not fall over on a typo typed at the prompt.

Three tiers per object:
  clean    11 paraphrases, including alternate specific names ("box of crackers")
  generic   3 superordinate phrasings ("pick up the can") -- only where safe, see below
  noisy     6 misspellings / formatting slips

Properties that are load-bearing and should survive any edit here:

1. A generic term is only added where the scenes contain at most ONE object of that
   category. Verified by sampling ~48 episodes across all downloaded tasks: distractors
   are drawn from a shared pool (cracker box, mustard bottle, nivea tube, ...) but never
   duplicate a category, and no scene held two fruits. That is what makes "pick up the
   can" a safe pointer to the tomato soup can.
   Deliberately NOT given generics:
     - "the cube": red-cube scenes contain a purple cube as the distractor, so two cubes
       are on the table at once.
     - "the cylinder": mugs are sometimes rendered as handleless cylinders and the soup
       can is cylindrical, so it points at three different things.
2. The five fruit pools SHARE the "fruit" generics on purpose -- the same sentence maps
   to a lemon in one scene and a pear in another, which is exactly the category
   generalization we want, and is unambiguous because only one fruit is ever present.
   The uniqueness guard at the bottom permits sharing ONLY for generic strings.
3. No *added* color or size adjectives on the GraspNet objects. They would be a second,
   unverified claim about the scene; the rendered lemon is a dull olive-yellow rather
   than the obvious "yellow", and a sample of frames is not evidence about thousands of
   episodes. The Franka_3_objects_2 objects are the exception and keep their colors,
   because there color IS the referent: those scenes put a purple cube next to the red
   cube and next to the orange cylinder, so "the cube" alone is genuinely ambiguous.
4. Nothing in the orange-cylinder pool may reduce to a bare "orange". The merged task
   table contains both "pick up the orange" (the fruit) and "pick up the orange
   cylinder"; they never co-occur in a scene, but the prompt space must keep them apart.
   Same reason "pick up the orange fruit" exists on the fruit side.

All variants mean strictly "pick up". Nothing implies a place/move phase -- this data
has no such phase.

The canonical string is always index 0 of the clean list and is weighted heaviest, so
evaluation prompts using the original wording stay firmly in distribution.
"""

# Verb/phrasing templates applied to every object.
_TEMPLATES = (
    "pick up the {n}",
    "grasp the {n}",
    "grab the {n}",
    "lift the {n}",
    "pick the {n} up",
    "take the {n}",
    "get the {n}",
    "pick up the {n} from the table",
)

# Misspelling patterns applied to every object, on top of two per-object name typos.
# "pickup" (missing space) and a capitalized form are included deliberately: pi0.5 uses
# PaligemmaTokenizer, whose tokenize() only strips and replaces "_"/"\n" -- unlike the
# FAST tokenizers it does NOT lowercase, so case reaches the model and is worth training.
_NOISY_TEMPLATES = (
    "pickup the {n}",
    "pik up the {n}",
    "pick up {n}",
    "Pick up the {n}",
)

# Superordinate category terms. Objects sharing a key share these exact strings.
_GENERIC_TEMPLATES = ("pick up the {g}", "grasp the {g}", "lift the {g}")
_GENERIC_TERMS = ("box", "bottle", "can", "tube", "cup", "fruit")

# object name -> (3 extra clean phrasings, 2 misspelled forms of the name, generic key)
_OBJECTS: dict[str, tuple[tuple[str, ...], tuple[str, str], str | None]] = {
    "cracker box": (
        ("reach for the cracker box and lift it", "pick up the box of crackers", "grasp the box of crackers"),
        ("craker box", "crakcer box"),
        "box",
    ),
    "lemon": (
        ("reach for the lemon and lift it", "grasp and lift the lemon", "pick up the lemon fruit"),
        ("lemmon", "lemn"),
        "fruit",
    ),
    "mug": (
        ("reach for the mug and lift it", "grasp and lift the mug", "take hold of the mug"),
        ("mugg", "mg"),
        "cup",
    ),
    "mustard bottle": (
        ("reach for the mustard bottle and lift it", "pick up the bottle of mustard", "grasp the bottle of mustard"),
        ("mustrad bottle", "mustard botle"),
        "bottle",
    ),
    # "the orange fruit" is not filler: "orange" is the one object name in this set that
    # collides with a color word, and with "orange cylinder" below.
    "orange": (
        ("reach for the orange and lift it", "grasp and lift the orange", "pick up the orange fruit"),
        ("ornage", "orang"),
        "fruit",
    ),
    "peach": (
        ("reach for the peach and lift it", "grasp and lift the peach", "pick up the peach fruit"),
        ("peech", "pech"),
        "fruit",
    ),
    # "pair" and "plumb" below are real English words, not nonsense strings. They are the
    # typos a person actually makes for these two objects, and nothing else in this
    # closed object set claims either word.
    "pear": (
        ("reach for the pear and lift it", "grasp and lift the pear", "pick up the pear fruit"),
        ("paer", "pair"),
        "fruit",
    ),
    "plum": (
        ("reach for the plum and lift it", "grasp and lift the plum", "pick up the plum fruit"),
        ("plumb", "plumm"),
        "fruit",
    ),
    "nivea men face wash tube": (
        ("pick up the face wash tube", "grasp the nivea tube", "lift the tube of face wash"),
        ("nivia men face wash tube", "nivea men facewash tube"),
        "tube",
    ),
    "tomato soup can": (
        ("reach for the tomato soup can and lift it", "pick up the can of tomato soup", "grasp the soup can"),
        ("tomatoe soup can", "tomato soop can"),
        "can",
    ),
    # --- Franka_3_objects_2. Colors are kept: they are the referent, not decoration.
    # No generics here -- see property 1 in the module docstring.
    "orange cylinder": (
        ("reach for the orange cylinder and lift it", "grasp and lift the orange cylinder",
         "take hold of the orange cylinder"),
        ("orange cylender", "ornage cylinder"),
        None,
    ),
    "red cube": (
        ("reach for the red cube and lift it", "pick up the red block", "grasp the red block"),
        ("red cbue", "red cub"),
        None,
    ),
    "purple cube": (
        ("reach for the purple cube and lift it", "pick up the purple block", "grasp the purple block"),
        ("purpel cube", "purple cbue"),
        None,
    ),
}

# Sampling weights, as integer multiplicities in the pool the transform indexes into.
# Uniform sampling over every variant would make ~30% of training prompts misspelled,
# which is more corruption than robustness needs. These give roughly 65% specific /
# 14% generic / 21% noisy, with the untouched original string the single largest share.
CANONICAL_WEIGHT = 15
CLEAN_WEIGHT = 4
GENERIC_WEIGHT = 4
NOISY_WEIGHT = 3


def _build() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for name, (extra_clean, typos, generic_key) in _OBJECTS.items():
        assert generic_key is None or generic_key in _GENERIC_TERMS, (name, generic_key)
        clean = [t.format(n=name) for t in _TEMPLATES] + list(extra_clean)
        noisy = [f"pick up the {t}" for t in typos] + [t.format(n=name) for t in _NOISY_TEMPLATES]
        generic = [] if generic_key is None else [t.format(g=generic_key) for t in _GENERIC_TEMPLATES]
        assert len(clean) == 11 and len(noisy) == 6, (name, len(clean), len(noisy))
        assert len(set(clean + noisy + generic)) == len(clean) + len(noisy) + len(generic), (
            f"duplicate variant within {name}"
        )

        pool = [clean[0]] * CANONICAL_WEIGHT
        for v in clean[1:]:
            pool += [v] * CLEAN_WEIGHT
        for v in generic:
            pool += [v] * GENERIC_WEIGHT
        for v in noisy:
            pool += [v] * NOISY_WEIGHT
        out[clean[0]] = tuple(pool)
    return out


# canonical task string -> weighted pool of variants (repeats encode the weights).
PROMPT_VARIANTS: dict[str, tuple[str, ...]] = _build()

# Every string that is legitimately shared between objects: the superordinate terms.
GENERIC_VARIANTS: frozenset[str] = frozenset(
    t.format(g=g) for g in _GENERIC_TERMS for t in _GENERIC_TEMPLATES
)

# A non-generic variant must belong to exactly one object, or the augmentation would
# teach two contradictory groundings for the same sentence. Generic strings are exempt --
# they are shared on purpose (all five fruits answer to "pick up the fruit"). This is the
# guard that catches a future edit reintroducing a bare "pick up the orange" into the
# cylinder pool, or a "the cube" generic that two cubes on one table would make ambiguous.
_owner: dict[str, str] = {}
for _canonical, _pool in PROMPT_VARIANTS.items():
    for _v in set(_pool):
        if _v in GENERIC_VARIANTS:
            continue
        if (_prev := _owner.setdefault(_v, _canonical)) != _canonical:
            raise AssertionError(f"variant {_v!r} claimed by both {_prev!r} and {_canonical!r}")
del _owner, _canonical, _pool, _v
