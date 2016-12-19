

import xml.etree.ElementTree as ET
from math import sqrt

def bb_from_xy(x1, y1, x2, y2):
    x = int((x1 + x2)/2.)
    y = int((y1 + y2)/2.)
    d = int(sqrt((x1-x2)**2 + (y1-y2)**2))
    return (x-2*d,  y-2*d, 3*d, 5*d)

def bb_from_gt(groundtruth_xml_path):
    f = open(groundtruth_xml_path, 'r')
    data = f.read()
    f.close()
    
    root = ET.fromstring(data) 
    print(root)
    #print([el.attrib.get('id') for el in tree.findall('person')])

    frame_dict = {}
    
    for frame in root.findall('./frame'):
        frame_num = frame.attrib.get('number')
        
        persons = []
        for person in frame.findall('./person'):            
            if person is None: continue
            
            person_id = person.attrib.get('id')
            
            leftEye = person.find('leftEye')
            xl, yl = int(leftEye.attrib.get('x')), int(leftEye.attrib.get('y'))
            
            rightEye = person.find('rightEye')
            xr, yr = int(rightEye.attrib.get('x')), int(rightEye.attrib.get('y'))

            bb = bb_from_xy(xl, yl, xr, yr)
            
            persons.append((person_id,  bb))
            
            print(person_id, frame_num, bb)

        if persons:
            frame_dict[frame_num] = persons
        
    return frame_dict

pathGT = './in/groundtruth/P1E_S1_C2.xml'

frames = bb_from_gt(pathGT)
print(frames)

#data = """your xml here"""

