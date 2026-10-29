# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Room-specific contents with explicit compartment bounds and independent bodies."""
import json

from scene_lib.bear_loop_furnishings import native

# Source identity, physical dimensions [m], mass [kg]. Different assets retain their source materials.
KITCHEN = [
    ('CanFactory_1526963__spawn_asset_5470950_', (.075,.075,.12), .4),
    ('FoodBagFactory_6129003__spawn_asset_9279215_', (.18,.12,.055), .5),
    ('BowlFactory_974736__spawn_asset_581637_', (.15,.15,.085), .25),
    ('FoodBoxFactory_9173839__spawn_asset_5983640_', (.13,.08,.21), .45),
    ('BottleFactory_1804529__spawn_asset_134498_', (.075,.075,.26), 1.05),
    ('CupFactory_9054484__spawn_asset_3849534_', (.105,.085,.10), .22),
    ('FoodBagFactory_6193854__spawn_asset_649510_', (.17,.12,.06), .12),
    ('CanFactory_1526963__spawn_asset_5470950_', (.095,.095,.07), .22),
    ('BottleFactory_1021896__spawn_asset_7412270_', (.065,.065,.18), .45),
    ('BowlFactory_974736__spawn_asset_549778_', (.21,.21,.12), .4),
    ('FoodBoxFactory_2301894__spawn_asset_7759540_', (.16,.075,.23), .35),
    ('KnifeFactory_9767186__spawn_asset_883869_', (.24,.032,.014), .08),
    ('CanFactory_1526963__spawn_asset_5470950_', (.06,.06,.15), .3),
    ('h48:bedroom_saucer_plate_0_s0_0', (.18,.18,.025), .2),
]
BATH = [
    ('h48:bathroom_bar_of_soap_0_s0_1',(.075,.05,.03),.1),
    ('h48:bathroom_shower_bottle_0',(.08,.06,.15),.3),
    ('h48:bathroom_toothbrush_tumbler_cup_0',(.08,.08,.10),.15),
    ('h48:bathroom_shower_bottle_1',(.065,.055,.17),.24),
    ('h48:bathroom_soap_dish_0_s0_0',(.12,.09,.03),.2),
    ('h98:bedroom_perfume_bottle_0',(.055,.04,.085),.12),
]
BED = [
    ('h98:bedroom_smartphone_0',(.075,.15,.012),.18),
    ('h47:bedroom_ballpoint_pen_0',(.14,.012,.012),.02),
    ('h48:bedroom_notebook_0',(.16,.21,.018),.22),
    ('h48:closet_cologne_bottle_0',(.07,.05,.10),.15),
    ('h47:bedroom_tv_remote_control_0',(.055,.16,.022),.1),
    ('h98:bedroom_hardcover_book_2_s0_0',(.15,.20,.03),.3),
]
CLOTHES = [
    ('h48:closet_knit_sweater_1_0',(.24,.23,.08),.4),
    ('h48:closet_denim_jeans_0',(.23,.24,.065),.55),
    ('h48:closet_wide_brim_hat_0',(.24,.24,.12),.2),
    ('h48:closet_knit_sweater_1_1_s0_0',(.24,.23,.07),.35),
]
BOOKS = [(f'h98:living_room_{name}',(.16,.21,.025),.3) for name in
         ('art_book_0','comic_book_0','reference_book_0','paperback_book_0','textbook_0','hardcover_book_0')]


def populate_compartments(house) -> None:
    """Place two distinct, fitting objects on every storage level with clearance [m]."""
    stores = house.storage_compartments
    supports = {s['support_path'] for s in stores} | {s['name'] for s in stores}
    old = {p for p in house.stored_items if house.stage.GetPrimAtPath(p).GetCustomDataByKey('support') in supports}
    for path in old:
        house.stage.RemovePrim(path)
    house.stored_items = [p for p in house.stored_items if p not in old]
    house.inventory = [r for r in house.inventory if r.get('path') not in old]
    counters = {}
    records = []
    for storage in stores:
        name = storage['name']
        if name.startswith('bath'):
            group, pool = 'bathroom', BATH
        elif name.startswith(('closet','dressing')):
            group, pool = 'clothing', CLOTHES
        elif name.startswith(('kitchen','pantry','wet_bar','refrigerator')):
            group, pool = 'kitchen', KITCHEN
        elif 'nightstand' in name or name.startswith('entry'):
            group, pool = 'bedside', BED
        else:
            group, pool = 'books', BOOKS
        levels = [(storage['support'], storage['height']-.025)] if storage['drawer'] else [
            (.025, storage['height']*.48-.0125),
            (storage['support'], storage['height']-.025),
        ]
        for level, (base, ceiling) in enumerate(levels):
            available = ceiling-base-.015
            fitting = [item for item in pool if item[1][2] < available and item[1][0] < storage['width']*.43]
            if len(fitting) < 2:
                raise ValueError(f'Not enough fitting distinct contents: {name}/{level}')
            start = counters.get(group,0)
            chosen = [fitting[(start+i)%len(fitting)] for i in range(2)]
            counters[group] = start+2
            if name == 'kitchen_island' and level == 0:
                chosen[1] = ('BottleFactory_1021896__spawn_asset_7412270_', (.05,.05,.15), .25)
            items = []
            for index,(key,size,mass) in enumerate(chosen):
                if size[2] >= available or size[0] >= storage['width']*.43:
                    raise ValueError(f'Contents exceed compartment clearance: {name}/{level}/{index}')
                dx = (-1 if index==0 else 1)*storage['width']*.24
                dy = -.035 if index==0 else .045
                pos = storage['world']((dx,dy,base))
                item_name = f'{name}_compartment_{level}_item_{index}'
                if ':' in key:
                    path = house.asset(item_name,key,pos,size,mass,support=storage['support_path'])
                else:
                    kind = 'sdf' if key.startswith(('Bowl','Cup')) else 'box'
                    if key.startswith('Bottle'):
                        kind = 'convexHull'
                    path = native(house,item_name,key,pos,size,mass,storage['support_path'],kind)
                house.stage.GetPrimAtPath(path).SetCustomDataByKey('compartment',f'{name}/{level}')
                house.stored_items.append(path)
                items.append(dict(path=path,source=key,size_m=size))
            records.append(dict(name=f'{name}/{level}',category=group,base_m=base,ceiling_m=ceiling,items=items))
    (house.metadata_dir/'compartments.json').write_text(json.dumps(records,indent=2)+'\n')
