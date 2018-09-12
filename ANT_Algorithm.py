import sys
import math
import numpy as np
import timeit
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.csgraph import shortest_path

#инициализируем все параметры
ALPHA = 1
BETA = 1
RHO = 0.5   # Коэффициент испарения феромонов за 1 шаг
iterations = 500  # Максимальное количество итераций
Pheromon_default = 100  # Коэффициент распыления феромона
Time_limit = 14.5  # Максимальное время выполнения программы, после окончания автоматически выдает последний наименьший результат

# старт таймера
list_el = [el for el in sys.stdin]  
timer_start = timeit.default_timer()

# парсим все строки 
n = np.asscalar(np.fromstring(list_el[0], dtype=int, sep=' '))

M = []
for el in list_el[1:n+1]:
    M.append(np.fromstring(el, sep=' '))
M = np.array(M)

terminal_way = 0

# осуществляем рекурсивный поиск и удаление терминальных узлов, все расстояния до терминальных нод складываются и учитываются в конце
while True:
    p = []
    # ищем только терминальные ноды, соответственно считаем их растояние и удаляем
    for i in range(len(M[0])):
        if np.count_nonzero(M[i])==1:
            p.append(i)
            terminal_way += M[i][np.nonzero(M[i])]*2
    M = np.delete(M, p, axis=0)
    M = np.delete(M, p, axis=1)
    if len(p)==0:   # Повторяем до тех пор пока не удалим все терминальные ноды
        break
M = np.copy(M, order='C')


    
n = len(M[0])


BFS_matrix = []
for i in range(n): # Ищем в ширину доступные города, для каждого города, в пределах 1 перехода
    city_list = []
    P, T = breadth_first_order(M, i)
    for k in P:
        if T[k] == i:
            city_list.append(k)
    BFS_matrix.append(city_list)

#здесь стоит быть аккуратным т.к. связность графа должны быть равна 1
SP_matrix, PredSP_Matrix = shortest_path(M, return_predecessors=True) # Строим матрицу кратчайших путей из i в j
SPW_matrix = []
for i in range(n):
    for k in range(n):
        el = []
        j=k
        if i==j:
            # el.append(0)
            pass
        else:
            el.append(j)
            while True:
                if PredSP_Matrix[i][j]==-9999:
                    break
                el.append(PredSP_Matrix[i][j])
                j = PredSP_Matrix[i][j]
        el = el[::-1]
        SPW_matrix.append(np.array(el))
SPW_matrix = np.array(SPW_matrix)    
SPW_matrix = SPW_matrix.reshape((n, n)) # SPW это матрица в которой хранятся все пути из i в j даже при условии что между ними нет прямой связи

cities = list(np.arange(n)) # [0 1 2 ... n-1] Список городов

RHO_matrix = np.full((n, n), 1-RHO)  #  Заполняем матрицу RHO коэффицентами
Pheromon_matrix = np.full((n, n), 1)  # Заполняем матрицу феромонов

# Функция построения пути муравья на основе матрицы переходов, феромонов итд        
def ant(start_city):
    tabu_list = np.delete(cities, start_city)   # табу лист не является жестким
    attended_cities = []
    attended_cities = np.append(attended_cities, start_city).astype(int)
    current_city = start_city
    while(len(tabu_list)):  # пока все еще есть города в табу листе
        possible_ways = BFS_matrix[current_city]  # Получаем список всех соседних городов
        unattended_cities = []
        for city in possible_ways:  # Просматриваем все соседние ноды с котоырми есть прямая связь
            p = np.where(tabu_list == city)[0]
            if len(p):
                p = p.item()
                unattended_cities.append(tabu_list[p])  # Если нода рядом есть и её нет в табу листе, то записываем в список которые надо посетить
        if len(unattended_cities) > 0:  # Есть ли рядом ноды в прямой видимости, которые находятся в списке табу?
            mov_prob = []
            for city in unattended_cities: # Расчитываем вероятность пути
                    mov_prob.append(move_posbl(current_city, city))
            mov_prob = [el * (1 / np.sum(mov_prob)) for el in mov_prob]
            
            unattended_cities = np.asarray(list(unattended_cities))
            mov_prob = np.asarray(list(mov_prob)) 
            next_city = np.random.choice(len(unattended_cities), 1, p=mov_prob)     # Рандомно определеяем в какую ноду идти в соответсвии с их вероятностями
            next_city = unattended_cities[next_city]
            next_city = next_city.item()
            tabu_list = np.delete(tabu_list, np.where(tabu_list == next_city))  # Удаляем ноду из табу листа
            attended_cities = np.append(attended_cities, next_city)     # Добавляем ноду к списку посещенных
        else:
            # сюда попадаем только когда мы находимся в точке и рядом нет городов которые мы не посещали, здесь то и начинается магия
            mov_prob = []
            ways = []
            for city in tabu_list:  # Смотрим все оставшиеся ноды из табу листа которые надо посетить, подробно считаем все вероятности 
                way = SPW_matrix[current_city][city]
                ways.append(way)
                mov = 0
                for dist in range(len(way)-1):
                    mov += move_posbl(way[dist], way[dist+1])
                mov_prob.append(mov)
            mov_prob = [el * (1 / np.sum(mov_prob)) for el in mov_prob] # Рандомно выбираем путь через несколько нод в соответствии с посчитаной привлекательностью
            next_city_index = np.random.choice(len(tabu_list), 1, p=mov_prob)
            next_city_index = next_city_index.item()
            next_city = tabu_list[next_city_index]
            for city in ways[next_city_index]:  # Удаляем все ноды через которые мы прошли
                tabu_list = np.delete(tabu_list, np.where(tabu_list == city))
            attended_cities = np.append(attended_cities, ways[next_city_index][1:]) # Удаляем все ноды из списка табу, и добавляем их к пройденому пути
        if len(tabu_list)==0:
            if next_city !=  start_city:    # Текущий город не является стартовым?
                tabu_list = np.append(tabu_list, start_city)    # Мы посетили все города из табу листа, осталось вручную добавить в него стартовый город
        current_city = next_city
    return attended_cities

# Функция расчитывающая привлекательность перехода из города start_city в destination_city
# Внимание! фнукция не осуществляет деление верояности каждого перехода на сумму всех предыдущих, т.к. в контектсте данной задачи этого не требуется
def move_posbl(start_city, destination_city):
    
    return np.power(Pheromon_matrix[start_city, destination_city], ALPHA)*np.power(1/M[start_city, destination_city], BETA)

# Фукнция для расчет пути, на вход подается массив из точек маршрута, на выходе выдается длина этого маршрута
def estm_dist(path):
    dist = 0
    for i in range(len(path)-1):
        dist += M[path[i]][path[i+1]]
    return dist

# Обновляем феромон на всем пути муравья исходя из пройденой им дистанции
def spray_pheromone(path, dist):
    for i in range(len(path)-1):
        Pheromon_matrix[path[i]][path[i+1]] += Pheromon_default / dist
        Pheromon_matrix[path[i+1]][path[i]] = Pheromon_matrix[path[i]][path[i+1]]
    return None

# Испаряем феромон, если феромон ниже нужного уровня то делаем coerce    
def evaporate_pheromone():
    global Pheromon_matrix
    Pheromon_matrix = Pheromon_matrix*RHO_matrix
    for i in range(n):
        for k in range(n):
            if Pheromon_matrix[i][k] < 1:
                Pheromon_matrix[i][k] = 1
    return None

# Основной цикл программы
min_dist = float('Inf')
dist_list = []
min
for i in range(iterations):     # Выполняем все итерации
    Path = []
    Dist = []
    for k in range(n):
        Path.append(ant(k))     # Раставляем муравьев в каждый город и запускаем их
    Dist = [estm_dist(el) for el in Path]   # Запоминаем маршруты движения и считаем их длины
    evaporate_pheromone()   # Испаряем феромон
    for k in range(n):
        spray_pheromone(Path[k], Dist[k])   # Для каждого маршрута наносим феромон

    min_estm_dist = np.amin(Dist)   # Смотрим самый корокий маршрут, обновляем инфомрацию о самом коротком маршруте
    if min_estm_dist < min_dist:
        min_dist = min_estm_dist
        dist_list.append(min_estm_dist)
    if (timeit.default_timer()- timer_start) > Time_limit:  # Смотрим на время выполнения скрипта, если время вышло то выходим из скрипта
        break
print(int(min_dist+terminal_way))   # Складываем ответ с суммой длин удаленных терминальных узлов
