
# Diningroom, single room only, first person view (~8min CPU runtime)
mkdir -p outputs/indoors/coarse-dining &
python scripts/generate_indoors.py --seed 0 --task coarse --output_folder outputs/indoors/coarse-dining -g fast_solve.gin singleroom.gin -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"DiningRoom\"\] &> outputs/indoors/coarse-dining/logs.log &

# Bathroom, single room only, first person view (~13min CPU runtime)
mkdir -p outputs/indoors/coarse-bathroom &
python scripts/generate_indoors.py --seed 0 --task coarse --output_folder outputs/indoors/coarse-bathroom -g fast_solve.gin singleroom.gin -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"Bathroom\"\] &> outputs/indoors/coarse-bathroom/logs.log &

# Bedroom, single room only, first person view (~10min CPU runtime)
mkdir -p outputs/indoors/coarse-bedroom &
python scripts/generate_indoors.py --seed 0 --task coarse --output_folder outputs/indoors/coarse-bedroom -g fast_solve.gin singleroom.gin -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"Bedroom\"\] &> outputs/indoors/coarse-bedroom/logs.log &

# Kitchen, single room only, first person view (~10min runtime, CPU only)
mkdir -p outputs/indoors/coarse-kitchen &
python scripts/generate_indoors.py --seed 0 --task coarse --output_folder outputs/indoors/coarse-kitchen -g fast_solve.gin singleroom.gin -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"Kitchen\"\] &> outputs/indoors/coarse-kitchen/logs.log &

# LivingRoom, single room only, first person view (~11min runtime, CPU only)
mkdir -p outputs/indoors/coarse-livingroom &
python scripts/generate_indoors.py --seed 0 --task coarse --output_folder outputs/indoors/coarse-livingroom -g fast_solve.gin singleroom.gin -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"LivingRoom\"\] &> outputs/indoors/coarse-livingroom/logs.log &

# Floor layout, overhead view, no objects (~34 second runtime, CPU only):
mkdir -p outputs/indoors/coarse-noobj-overhead &
python scripts/generate_indoors.py --seed 0 --task coarse --output_folder outputs/indoors/coarse-noobj-overhead -g no_objects.gin overhead.gin -p compose_indoors.terrain_enabled=False &> outputs/indoors/coarse-noobj-overhead/logs.log &

# Single random room with objects, overhead view (~11min. runtime CPU only):
mkdir -p outputs/indoors/coarse-overhead &
python scripts/generate_indoors.py --seed 0 --task coarse --output_folder outputs/indoors/coarse-overhead -g fast_solve.gin overhead.gin singleroom.gin -p compose_indoors.terrain_enabled=False compose_indoors.overhead_cam_enabled=True restrict_solving.solve_max_rooms=1 compose_indoors.invisible_room_ceilings_enabled=True compose_indoors.restrict_single_supported_roomtype=True &> outputs/indoors/coarse-overhead/logs.log  &

# Whole apartment with objects, overhead view:
mkdir -p outputs/indoors/coarse-apartment &
python scripts/generate_indoors.py --seed 0 --task coarse --output_folder outputs/indoors/coarse-apartment -g fast_solve.gin overhead.gin -p compose_indoors.terrain_enabled=False &> outputs/indoors/coarse-apartment/logs.log &

wait