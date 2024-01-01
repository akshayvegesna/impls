class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """

        n = len(matrix)

        for i in range(n): 
            for j in range(0, i): 
                # swap i, j with j, i 
                tmp = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = tmp 
        
        for i in range(n): 
            matrix[i] = matrix[i][::-1]
        
        return matrix        
        


if __name__ == '__main__': 
    s = Solution()
    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    print(s.rotate(matrix))
    matrix = [[1,2],[3,4]]
    print(s.rotate(matrix))
    