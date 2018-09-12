# ACO_Algorithm
Решение задачи коммивояжера на языке Python

Основные идеи решения задачи
=====================
1) В основе используется решение предложенное в статье [Штовба С.Д. Муравьиные алгоритмы](https://www.researchgate.net/publication/279535061/download)
-----------------------------------
2) Реализован алгоритм автоматического поиска и удаление терминальных узлов графа. Это приводит к тому, что реальный путь в ответе теряется, однако на конечный результат это никак не влияет. В реальных задачах очень часто встречается большое количество терминальных узлов, подсчет и удаление которых может значительно повысить производительность алгоритма.
-----------------------------------
3) Предварительно происходит расчет вспомогательных матриц, а именно матрица для каждого элемента с областью видимости в 1 переход и матрица в которой рассчитаны кратчайшие пути из любого i в любой j узел даже если ними нет прямой связи, но при условии связности графа равной 1. Данные вспомогательные матрицы помогает значительно ускорить алгоритм в случае мягкого списка табу.
-----------------------------------
4) Почти все реализации Муравьиных алгоритмов, представленных в интернете, имеют т.н. жесткий табу лист, т.е. по условию муравью запрещено посещать уже посещенные города, однако это приводит к проблемам, когда в графе присутствуют узлы, не имеющие прямой связи. В связи с этим в данном алгоритме реализован т.н. мягкий табу лист, который позволяет строить оптимальный маршрут через посещенные города, если в прямой видимости нет городов из списка табу, при этом выбор также осуществляется исходя из подсчёта привлекательности этих маршрутов. В случае если в графе отсутствуют длинные переходы, то данный алгоритм автоматически вырождается в алгоритм с жестким табу листом, однако в обратном случае не теряет свою функциональность. 
-----------------------------------
5) Алгоритм снабжен двумя условиями выхода из главного цикла - количество итераций и время исполнения.
-----------------------------------
