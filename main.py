#!/usr/bin/env python

# Some code is inspired by by Carla examples written by German Ros (german.ros@intel.com)

from __future__ import print_function

LEAVE_TRACES = False
DETECT_OBJECTS = True
CONCATENATE_MINIMAP = True

import collections
import datetime
import glob
import math
import os
import random
import re
import sys
import weakref
import time

import cv2 # OpenCV

# Solution for classic TSP problem
# https://github.com/fillipe-gsm/python-tsp
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.heuristics import solve_tsp_local_search

# Other options
# https://developers.google.com/optimization/routing/tsp#program1

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

# TODO: cleanup these two
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

#
# Carla color objects - useful in debugging functions
#

red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)
black = carla.Color(0, 0, 0)

# ==============================================================================
# -- class RouteAgent ----------------------------------------------------------
# ==============================================================================

class RouteAgent(BasicAgent):
    """
    RouteAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        super(RouteAgent, self).__init__(vehicle, target_speed)
        self._location = self._vehicle.get_location()

    """
    Main entry point to tell the agent to go over every street and every lane
    """
    def crawl_everything(self):
        locations = self._get_ordered_locations()
        self.set_destinations(locations)

    """
    Status checking to see if there are destinations left
    """
    def has_destinations(self):
        return self._destination_id < len(self._destinations)

    def set_destinations(self, destinations):
        n = len(destinations)
        print(f"Have {n} destinations to go to")
        self._destinations = destinations
        self._destination_id = 0

        time = datetime.datetime.now().strftime("%H:%M:%S")
        self._status = f"{time}: going to waypoint {self._destination_id} of {len(self._destinations)}"
        print(self._status)
        d = self._destinations[self._destination_id]
        self.set_destination((d.x, d.y, d.z))

    def status(self):
        return self._status

    def run_step(self, debug=False):
        control = super(RouteAgent, self).run_step(debug)

        location = self._vehicle.get_location()
        if LEAVE_TRACES:
            # draw a tracer line along the vehicle path
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self._world.debug.draw_line(self._location, location, 1, carla.Color(r, g, b))
        self._location = location

        if not self._local_planner.done():
            return control  # continue

        time = datetime.datetime.now().strftime("%H:%M:%S")
        self._destination_id += 1
        if self._destination_id >= len(self._destinations):
            print(f"{time}: done with the destination list")
            return control # we are done

        time = datetime.datetime.now().strftime("%H:%M:%S")
        self._status = f"{time}: going to waypoint {self._destination_id} of {len(self._destinations)}"
        print(self._status)
        d = self._destinations[self._destination_id]
        self.set_destination((d.x, d.y, d.z))

        # run the step again. Maybe better to just figure out the control?        
        control = super(RouteAgent, self).run_step(debug)
        return control

    def get_minimap_image(self, template):
        """
        Returns the mini-map image with the route layout, next waypoint,
        current position and current overall status
        """
        image = np.zeros_like(template)
        H, W = image.shape[:2]

        seed = self._destinations[0]
        
        carla_min_x = seed.x
        carla_max_x = seed.x
        carla_min_y = seed.y
        carla_max_y = seed.y

        for i in range(len(self._destinations)):
            d = self._destinations[i]
            if d.x < carla_min_x:
                carla_min_x = d.x
            if d.x > carla_max_x:
                carla_max_x = d.x
            if d.y < carla_min_y:
                carla_min_y = d.y
            if d.y > carla_max_y:
                carla_max_y = d.y

        margin = 20  # mixels off the edge
        draw_min_x = margin
        draw_max_x = W - margin
        draw_min_y = margin
        draw_max_y = H - margin

        def map_point(carla_x, carla_y): # carla -> draw mapping
            draw_x = draw_min_x + int((draw_max_x - draw_min_x) / (carla_max_x - carla_min_x) * (carla_x - carla_min_x))
            draw_y = draw_min_y + int((draw_max_y - draw_min_y) / (carla_max_y - carla_min_y) * (carla_y - carla_min_y))

            return draw_x, draw_y

        for i in range(len(self._destinations)-1, 0, -1):
            d = self._destinations[i]

            # draw point
            d_x, d_y = map_point(d.x, d.y)
            color = (255, 255, 255) # white
            size = 3
            if i == self._destination_id:
                color = (0, 255, 255) # yellow
                size = 8
            elif i < self._destination_id:
                color = (0, 255, 0) # green
            
            cv2.circle(image, (d_x, d_y), size, color, size)

            # draw segment
            if i < len(self._destinations) - 2:
                dp = self._destinations[i-1]
                dp_x, dp_y = map_point(dp.x, dp.y)
                cv2.line(image, (dp_x, dp_y), (d_x, d_y), color, 2)

        # draw vehicle
        loc = self._vehicle.get_location()
        v_x, v_y = map_point(loc.x, loc.y)
        cv2.circle(image, (v_x, v_y), 10, (0, 0, 255), size)

        return image

    #
    # Helper methods
    #

    def _draw_topology(self):
        map = self._world.get_map()
        waypoint_tuple_list = map.get_topology()
        
        colors = [red, green, blue, black, yellow, cyan, orange]
        c = 0

        for tuple in waypoint_tuple_list:
            loc1 = tuple[0].transform.location
            loc2 = tuple[1].transform.location
            # print(loc1, loc2)
            # self._world.debug.draw_point(loc1)
            # self._world.debug.draw_point(loc2)
            color = colors[c % len(colors)]
            c += 1
            self._world.world.debug.draw_line(loc1, loc2, 1.5, color)

    def _draw_edges(self, locations, edges):
        colors = [red, green, blue, black, yellow, cyan, orange]
        c = 0

        for edge in edges:
            loc1 = locations[edge["start"]]
            loc2 = locations[edge["end"]]

            color = colors[(c//10) % len(colors)]
            c += 1
            self._world.debug.draw_line(loc1, loc2, 1.5, color)

    def _draw_route(self, route):
        colors = [black, red]
        c = 0

        for i in range(len(route)-1):
            time.sleep(0.5)
            loc1 = route[i]
            loc2 = route[i+1]

            # distance = self._get_crowflight_distance(loc1, loc2)
            # print(distance)

            color = colors[c % len(colors)]
            c += 1
            self._world.debug.draw_line(loc1, loc2, 3, color)

        return

    def _get_topology_waypoints(self):
        id2wp = {}
        waypoints = []

        topology = self._map.get_topology()
        for wp1, wp2 in topology:
            if not wp1.id in id2wp:
                id2wp[wp1.id] = wp1
                waypoints.append(wp1)
            if not wp2.id in id2wp:
                id2wp[wp2.id] = wp2
                waypoints.append(wp2)

        return waypoints

    def _get_crowflight_distance(self, loc1, loc2):
        return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)

    def _get_manhattan_distance(self, loc1, loc2):
        return abs(loc1.x - loc2.x) + abs(loc1.y - loc2.y)

    # TODO: refactor router call
    def _get_route_distance(self, router, loc1, loc2):
        route = router.trace_route(carla.Location(loc1.x, loc1.y, 0.0), carla.Location(loc2.x, loc2.y, 0.0))
        return len(route)  # distance in the number of hops

    def _get_distance(_self, router, loc1, loc2):
        return self._get_route_distance(router, loc1, loc2)

    def _are_close(self, loc1, loc2):
        distance = self._get_crowflight_distance(loc1, loc2)
        return distance < 1

    # TODO: cleanup
    def _get_ordered_locations_topology(self):
        topo = self._map.get_topology()

        # TODO: make more efficient
        locations = []
        edges = []

        def find_close_location(loc):
            for i in range(len(locations)):
                if self._are_close(locations[i], loc):
                    return i

            return -1

        def edge_exists(start, end):
            for edge in edges:
                if edge["start"] == start and edge["end"] == end:
                    return True

            return False

        for wp1, wp2 in topo:
            loc1 = wp1.transform.location
            index1 = find_close_location(loc1)
            if index1 < 0:
                locations.append(loc1)
                index1 = len(locations)-1

            loc2 = wp2.transform.location
            index2 = find_close_location(loc2)
            if index2 < 0:
                locations.append(loc2)
                index2 = len(locations)-1

            if not edge_exists(index1, index2):
                edges.append({
                    "start": index1,
                    "end":   index2
                })

        # self._draw_edges(locations, edges)

        # print(locations)
        #print(edges)

        # cleanup - use the built-in router
        hop_resolution = 2.0
        dao = GlobalRoutePlannerDAO(self._map, hop_resolution)
        router = GlobalRoutePlanner(dao)
        router.setup()

        distance_cache = {}  # (from_x, from_y, to_x, to_y) -> distance

        ordered_index = []
        while len(edges) > 0:
            # find an edge with the start closest to the last point
            if len(ordered_index) == 0:
                # first location - we can start with the current location of the vehicle
                loc_last = self._vehicle.get_location()
            else:
                loc_last = locations[ordered_index[len(ordered_index)-1]]

            # TODO: extract as a function
            best_edge_index = None
            shortest_distance = None
            for edge_index in range(len(edges)):
                loc_start_index = edges[edge_index]["start"]
                loc_start = locations[loc_start_index]
                if (loc_last.x, loc_last.y, loc_start.x, loc_start.y) in distance_cache:
                    distance = distance_cache[(loc_last.x, loc_last.y, loc_start.x, loc_start.y)]
                else:
                    distance = self._get_route_distance(router, loc_last, loc_start)
                    distance_cache[(loc_last.x, loc_last.y, loc_start.x, loc_start.y)] = distance
                if best_edge_index is None or distance < shortest_distance:
                    shortest_distance = distance
                    best_edge_index = edge_index

            if len(ordered_index) == 0 or ordered_index[len(ordered_index)-1] != edges[best_edge_index]["start"]:
                ordered_index.append(edges[best_edge_index]["start"])
            if len(ordered_index) == 0 or ordered_index[len(ordered_index)-1] != edges[best_edge_index]["end"]:
                ordered_index.append(edges[best_edge_index]["end"])
            del edges[best_edge_index]
            l = len(edges)
            print(f"Have {l} edges to process")

        end_s = time.time()

        ordered = []
        for index in ordered_index:
            ordered.append(locations[index])

       # self._draw_route(ordered)
        return ordered

    def _get_ordered_locations(self):
        # I experimented with other algorithms, but ultimate left this one
        return self._get_ordered_locations_topology()

# ==============================================================================
# -- ObjectDetector ------------------------------------------------------------
# ==============================================================================

class ObjectDetector:
    """
    Tutorial from:
    https://www.youtube.com/watch?v=HXDD7-EnGBY
    
    Training model from:
    https://github.com/ankityddv/ObjectDetector-OpenCV

    """
    def __init__(self):
        self._threshold = 0.65
        self._class_name = []
        self._class_file = 'coco.names'

        with open(self._class_file,'rt') as f:
            self._class_names = f.read().rstrip('\n').split('\n')

        self._config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self._weights_path = 'frozen_inference_graph.pb'

        self._net = cv2.dnn_DetectionModel(self._weights_path, self._config_path)
        self._net.setInputSize(320,320)
        self._net.setInputScale(1.0/ 127.5)
        self._net.setInputMean((127.5, 127.5, 127.5))
        self._net.setInputSwapRB(True)

    def markup(self, image):
        class_ids, confs, bbox = self._net.detect(image, confThreshold=self._threshold)
#        print(class_ids, bbox)

        image = image.astype(np.uint8)

        if len(class_ids) != 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                # some experimental class overrides - one cheap way to improve detection for Carla objects
                # if class_id == 1 or class_id == 10:
                #   class_id = 10  # traffic light
                # elif class_id == 12 or class_id == 13:
                #   class_id = 12  # street sign
                # else:
                #   continue

                cv2.rectangle(image, box, color=(0,255,0), thickness=2)
                cv2.putText(image, self._class_names[class_id-1].upper(),(box[0]+10,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

                # print(self._class_names[class_id-1].upper())

        return image

# ==============================================================================
# -- Camera --------------------------------------------------------------------
# ==============================================================================

class Camera:
    def __init__(self, world, vehicle):
        self._world = world
        self._vehicle = vehicle

        blueprint_library = self._world.world.get_blueprint_library()

        bp = blueprint_library.find('sensor.camera.rgb')
        transform = carla.Transform(carla.Location(x=2.0, z=2.4))
        # Experimentation with resolutions in the attempt to improve object recognition
        # bp.set_attribute('image_size_x', '1280')
        # bp.set_attribute('image_size_y', '1024')
        # bp.set_attribute('image_size_x', '1024')
        # bp.set_attribute('image_size_y', ' 768')
        # bp.set_attribute('sensor_tick', '0.1')
        bp.set_attribute('enable_postprocess_effects', 'False')

        self._camera = self._world.world.spawn_actor(bp, transform, attach_to=vehicle)

        print(f"Created {self._camera.type_id}")

        self._camera.listen(lambda image: self._process_image(image))

    def _process_image(self, image_object):
        image = np.array(image_object.raw_data)
        image = image.reshape((image_object.height, image_object.width, 4))
        prep = image[:, :, :3]
        
        self._image = prep

    def image(self):
        return self._image

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.player = None
        self.restart()
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        """Restart the world"""
        # print available vehicles - for the choice.
        # for v in self.world.get_blueprint_library().filter("vehicle.*"):
        #    print(v)
        blueprint = random.choice(self.world.get_blueprint_library().filter("vehicle.carlamotors.carlacola"))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        print("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point = spawn_points[42]
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- Main Loop ---------------------------------------------------------
# ==============================================================================

def main_loop(host, port):
    """ Main loop for agent"""

    world = None

    last_time_ms = time.time() * 1000

    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)

        carla_world = client.load_world('Town02_Opt', carla.MapLayer.All)
        # carla_world.unload_map_layer(carla.MapLayer.All)

        world = World(client.get_world())

        camera = Camera(world, world.player)
        object_detector = ObjectDetector()
        agent = RouteAgent(world.player, target_speed=20)
        agent.crawl_everything()

        while agent.has_destinations():
            # As soon as the server is ready continue!
            if not world.world.wait_for_tick(10.0):
                continue

            # as soon as the server is ready continue!
            world.world.wait_for_tick(10.0)

            now_ms = time.time() * 1000
            if now_ms - last_time_ms > 10:
                image = camera.image().copy()
                # cv2.putText(image, agent.status(), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                minimap = agent.get_minimap_image(image)
                if CONCATENATE_MINIMAP:
                    concatinated = cv2.hconcat([image, minimap])
                    cv2.imshow("View", concatinated)
                else:
                    cv2.imshow("Camera View", image)
                    cv2.imshow("Minimap", minimap)
                cv2.waitKey(1)
                last_time_ms = time.time() * 1000
 
            control = agent.run_step(False)

            control.manual_gear_shift = False
            world.player.apply_control(control)

    finally:
        if world is not None:
            world.destroy()

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def main():
    host = '127.0.0.1'
    port = 2000

    print(f"Connecting to server {host}:{port}")

    try:
        main_loop(host, port)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()
