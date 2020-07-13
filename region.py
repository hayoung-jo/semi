import shapely as sh
import numpy as np
from functools import reduce
class Region:
    def __init__(self,radius):
        self.region_dict={}
        self.radius=radius
        pass
    def create(self,data_set,y_db):
        """
        region을 생성 받음
        ---------------------------------------------------------------------------
        입력
            data_set: persistence의 birth,death 쌍의 데이터 셋 N x 2 배열
            y_db: persistence를 클러스터링한 결과 -1은 노이즈, 0부터 클러스터 숫자
        """
        data_set=np.array(data_set)
        set_label=set(y_db)
        for label in set_label:
            if label==-1:
                self.neg_list=data_set[y_db==label]
                continue
            self.region_dict[label]=self._make_polygon_region(data_set[y_db==label])
        pass
    def predict(self,persistences):
        """
        예측
        ---------------------------------------------------------------------------
        입력
            persistences: persistence들의 집합 샘플수 x persistence 수 x 2(x,y)로 구성

        """
        result=[]
        
        for persistence in persistences:
            sum_=0
            for key in self.region_dict.keys():
                #sum_+=self.region_dict[key].predict(persistence,self.radius)    
                return self.region_dict[key].predict(persistence,self.radius)
            return False
            #result.append(sum_)
        #tmp=reduce(lambda x, y : x+y,result)
        
        #return tmp


        pass

    def update(self,persistences):
        """
        region_dict의 영역을 업데이트함
        ---------------------------------------------------------------------------
        입력
            persistences: persistence들의 집합 샘플수 x persistence 수 x 2(betti, cord)로 구성
        """          
        pass
    def _make_polygon_region(self,data_set):
        """
        data_set을 감싸는 convex hull 반환

        """
        data_set=list(data_set)

        data_set.sort(key=lambda l:l[0])
        x_min=data_set[0][0]
        x_max=data_set[-1][0]
        data_set.sort(key=lambda l:l[1])
        y_min=data_set[0][1]
        y_max=data_set[-1][1]
        result=[]
        for x in [x_min,x_max]:
            for y in [y_min,y_max]:
                result.append([x,y])
                
        
        return Convex(result)
        pass



class Convex:
    def __init__(self,points):
        poly=sh.geometry.Polygon(points)
        self.polygon=poly.convex_hull
    def predict(self,cord,radius):
        point=sh.geometry.Point(cord)
        if type(self.polygon)==sh.geometry.polygon.Polygon:
            result=point.within(self.polygon)
            if result:
                return result
            result=self.polygon.exterior.contains(point)
            return result
        else:
            result=self.polygon.distance(sh.geometry.Point(cord))<radius
            return result
        '''
        if type(self.polygon)==sh.geometry.linestring.LineString:
            result=self.polygon.distance(cord) < 1e-8  # True
            return result
        if type(self.polygon)==sh.geogetry.point.Point:
            result=self.polygon.distance(cord) < 1e-8
            return resutl 
        '''