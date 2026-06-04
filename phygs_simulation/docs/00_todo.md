# TODO

We want a procedurally generated, interactable, and photorealistic environments.

* procedurally generated (infinigen)
    * examples (05/02)
        <details>
            <summary>overhead-blender</summary>
            <IMG src="assets/indoors/coarse-apartment.png"  alt="image.png"/>
        </details>
        <details>
            <summary>bathroom-blender</summary>
            <IMG src="assets/indoors/coarse-bathroom.png"  alt="image.png"/>
        </details>
        <details>
            <summary>bathroom-isaacsim-naive</summary>
            <IMG src="assets/blend_to_isaac/naive.png"  alt="image.png"/>
        </details>
        <details>
            <summary>bathroom-isaacsim-handtuned</summary>
            <IMG src="assets/blend_to_isaac/handtuned.png"  alt="image.png"/>
        </details>
        <details>
            <summary>more scenes with more views (05/09)</summary>
            <IMG src="assets/tiny_25views/0.gif"  alt="image.png"/>
            <IMG src="assets/tiny_25views/1.gif"  alt="image.png"/>
        </details>

    * object generation/placement
        * what does infinigen do? what do other eai benchmarks do?
            <details>
                <summary>25 handheld object categories in infinigen (05/09)</summary>
                <IMG src="assets/handheld/handheld.png"  alt="image.png"/>
            </details>
        * floorplans (05/09)
            <details>
                <summary>extract object dimensions and locations from blender</summary>
                "StandingSinkFactory(4337611).spawn_placeholder(3020338)": {</br>
                    "type": "MESH",</br>
                    "dimensions": [</br>
                    0.5668395161628723,</br>
                    0.7741512060165405,</br>
                    0.72685706615448</br>
                    ],</br>
                    "location": [</br>
                    3.723379373550415,</br>
                    1.6784826517105103,</br>
                    0.655614972114563</br>
                    ],</br>
                    "rotation_euler": [</br>
                    0.0,</br>
                    0.0,</br>
                    -3.141592502593994</br>
                    ]</br>
                },</br>
                ...
            </details>
            <details>
                <summary>rendered floorplan</summary>
                <IMG src="assets/tiny_floorplan/0_floorplan.png"  alt="image.png"/>
                <IMG src="assets/tiny_floorplan/0_render.png"  alt="image.png"/>
            </details>
        * which objects
            * predefined objects from YCB/GSO?
        * language description? instance label?
        * where
        * how many
        * a separate object placement pipeline after the infinigen scenes are generated
            * up to discussion whether we keep the infinigen assets or generated object
            * *spatial feasilbity* continue exposing parameters, maybe boolean subtraction of mesh possible for empty space recongition
            * *semantic feasibility* clusters, rooms, furniture
    
    * articulation
        * in blender
            <details>
                <summary>simple_cabinets (05/02)</summary>
                <IMG src="assets/simple_cabinets/000000.gif"  alt="image.png"/>
                <IMG src="assets/simple_cabinets/000001.gif"  alt="image.png"/>
            </details>
            <details>
                <summary>simple_cabinet parameter export (05/09)</summary>
                {<br>
                    "shelf": {<br>
                        "Dimensions": [<br>
                            0.30488135039273245,<br>
                            0.5860757465489678,<br>
                            1.4424870384644795<br>
                        ],<br>
                        "bottom_board_height": 0.083,<br>
                        "shelf_depth": 0.29488135039273244,<br>
                        "shelf_cell_height": [<br>
                            0.3398717596161199,<br>
                            0.3398717596161199,<br>
                            0.3398717596161199,<br>
                            0.3398717596161199<br>
                        ],<br>
                        "shelf_cell_width": [<br>
                            0.5860757465489678<br>
                        ],<br>
                        "frame_material": "black_wood"<br>
                    },<br>
                    "door": {<br>
                        "door_width": 0.3100368788333984,<br>
                        "num_door": 2,<br>
                        "door_height": 1.4665120446814333,<br>
                        "frame_material": null,<br>
                        "door_left_hinge": true,<br>
                        "edge_thickness_1": 0.013537073673776953,<br>
                        "edge_width": 0.03556256505849167,<br>
                        "edge_thickness_2": 0.005989759277894055,<br>
                        "edge_ramp_angle": 0.7916421244694045,<br>
                        "board_thickness": 0.008537073673776954,<br>
                        "knob_R": 0.005791928675018841,<br>
                        "knob_length": 0.018826525114771463,<br>
                        "attach_height": [<br>
                            0.07629203209680452,<br>
                            1.3902200125846287<br>
                        ],<br>
                        "has_mid_ramp": false,<br>
                        "panel_material": [<br>
                            null<br>
                        ]<br>
                    }<br>
                }<br>
            </details>
        * how in isaac sim?
* interactable (isaacsim)
    * examples (05/09)
    * robots (tested in isaacsim and infinigen)
        * observations (done)
        * arm actions (snap-grasp done, no continuous control)
        * leg actions (teleport done, no continuous control)
    * objects (tested in isaacsim, not with infinigen)
        * pick/place (still need verification for infinigen scenes/objects)
        * collision
        * mass
    * articulation (not tested in isaacsim or infinigen)
        * open/close
* photorealistic (infinigen & isaacsim)
    * lack of lights in generation (yue: lighting is controllable)
    * object textures look washed when exported to usd
    * external spatial lights broken in isaacsim