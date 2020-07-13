
from region import Region
import gudhi
from sklearn.cluster import DBSCAN
from functools import partial,reduce
import numpy as np
import sys

class Classifier:
    def __init__(self,dim=2,delay=1,skip=1,simplex=gudhi.AlphaComplex,inf_value=999999):
        """
        분류기 초기값 설정
        """
        self.inf=sys.float_info.max/1e30
        ###
        self.region_dict={}
        ###time embedding 설정
        self.te=gudhi.point_cloud.TimeDelayEmbedding(dim=dim,delay=delay,skip=skip)
        ###simplex 설정
        self.simplex=simplex
        #DBSCAN설정
        self.db_dict={}
        self.db_params={}
        self.set_db_params()
        #self.db=DBSCAN(eps=eps,min_samples=min_samples,metric=metric,metric_params=metric_params,algorithm=algorithm,leaf_size=leaf_size,p=p,n_jobs=n_jobs)
        pass
    def set_db_params(self,index=-1,eps=0.5,min_samples=5,metric='euclidean',metric_params=None,algorithm='auto',leaf_size=30,p=None,n_jobs=None):
        self.db_params[index]={'eps':eps,'min_samples':min_samples,'metric':metric,'metric_params':metric_params,'algorithm':algorithm,'leaf_size':leaf_size,'p':p,'n_jobs':n_jobs}
    def get_db_params_index(self):
        return self.db_params.keys()
    def show_db_params(self,index):
        return self.db_params[index]     
        pass
    def train(self,data_set):
        """
        데이터 셋을 받아서 분류기 모델 설정
        ---------------------------------------------------------------------------
        입력
            data_set: 학습할 데이터 셋
        """
        ###make persistence
        persistence=self._make_persistence(data_set)
        ###make group region
        self._make_region(persistence)
        pass
    def predict(self,data_set):
        """
        입력 데이터 셋의 결과 도출
        ---------------------------------------------------------------------------
        입력
            data_set: 테스트 데이터 셋 1 x persistence 수 x 2(betti수, cord)
        """
        '''
        ###make persistence
        result=[]
        for persistence in data_set:
            persistences=self._make_persistence([persistence])
            self.predict_persistences=persistences
            ###predict data set
            data_dict_by_betti=self._arrange_persistences(persistences)
            persistence_result=[]
            for key in data_dict_by_betti.keys():
                persistence_result.append(self.region_dict[key].predict(data_dict_by_betti[key]))
            result.append(reduce(lambda x,y : x+y, persistence_result))
        '''
        result=[]

        self.predict_persistences=data_set
        data_dict_by_betti=self._arrange_persistences([data_set])
        persistence_result=[]
        for key in data_dict_by_betti.keys():
            persistence_result.append(self.region_dict[key].predict(data_dict_by_betti[key]))
        result=reduce(lambda x,y : x*y, persistence_result)
        print('result: ', result)
        
        return not result
        '''
        index=0

        #'''
        #return list(map(lambda x :int(not x),result))
        '''
        print(result)
        for r in result:
            set_r=set(r)
            for s in set_r:
                print(index,' num of ',s,' is ',r.count(s))
            index+=1
        print('\n\n')
        '''
        pass
    def update(self,data_set):
        """
        분류기 모델에 추가 학습
        ---------------------------------------------------------------------------
        입력
            data_set: 추가할 데이터 셋
        """
        persistences=self._make_persistence(data_set)
                ###betti수에 맞게 데이터 persistence를 묶음
        data_dict_by_betti={}
        for persistence in persistences:
            for betti, cord in persistence:
                if betti not in data_dict_by_betti.keys():
                    ##
                    data_dict_by_betti[betti]=[]
                if betti not in self.region_dict.keys():
                    self.region_dict[betti]=Region(self.db_params[self._get_db_params_dict_key(betti)]['eps'])
                data_dict_by_betti[betti].append(cord)
        ###betti수에 맞는 region 생성
        for key in data_dict_by_betti.keys():
            self.region_dict[key].update(persistence)
        pass
    def __get_persistence(self,te,data):
        """
        ---------------------------------------------------------------------------
        입력
            te: time embedding 함수
            data:data set

        """
        embedding=te(data)
        simplex=self.simplex(embedding)
        tree=simplex.create_simplex_tree()
        persistence=tree.persistence()
        return persistence

    def _make_persistence(self,data_set):
        """
        데이터를 입력 받아서 time embedding과 persistence를 수행하고 결과를 리턴
        ---------------------------------------------------------------------------
        입력
            data_set: persistence를 만들 데이터 셋
        """
        func=partial(self.__get_persistence,self.te)
        persistences=list(map(func,data_set))
        return persistences
        pass
    def _make_region(self,persistences):
        """
        betti 수에 맞는 영역을 추출함
        """
        ###betti수에 맞게 데이터 persistence를 묶음
        data_dict_by_betti=self._arrange_persistences(persistences)
        ###betti수에 맞는 region 생성
        for key in data_dict_by_betti.keys():
            index=key
            if key not in self.db_params.keys():
                index=-1
            self.db_dict[key]=DBSCAN(**self.db_params[index])
            y_db=self.db_dict[key].fit_predict(data_dict_by_betti[key])
            self.region_dict[key].create(data_dict_by_betti[key],y_db)
            pass
    def _get_db_params_dict_key(self,key):
        if key in self.db_dict.keys():
            return key
        else:
            return -1
    def _arrange_persistences(self,persistences):
        """
        betti number를 key로 하고 persistence를 가지는 dictionary를 반환
        ---------------------------------------------------------------------------
        입력
            persistences: persistence들의 집합 샘플수 x persistence 수 x 2(betti, cord)로 구성
        """   
        data_dict_by_betti={}
        for persistence in persistences:
            for betti, cord in persistence:
                if betti not in data_dict_by_betti.keys():
                    data_dict_by_betti[betti]=[]
                if betti not in self.region_dict.keys():
                    self.region_dict[betti]=Region(self.db_params[self._get_db_params_dict_key(betti)]['eps'])
                cord=list(cord)
                if cord[1]==np.inf:
                    #continue
                    cord[1]=self.inf
                data_dict_by_betti[betti].append(cord)
        return data_dict_by_betti
        pass