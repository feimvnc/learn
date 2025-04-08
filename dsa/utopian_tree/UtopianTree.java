/*
The utopian tree grows through 2 cycles.
Each spring, it doubles in height.
Eash summer, its height grow 1 meter.

Example
year = 5, the height is:
year    height
0       1
1       2
2       3
3       6
4       7
5       14

Return the height of tree from input of year.

1,2,3,4,5,6,7       year (2 cycles per year)
[1,2][3,4][4,6][7,8]
 1,2, 3,6, 7,14,15     height
      3    7    15
      4-1 8-1  16-1
      2^2-1
          2^3-1
                2^4-1

Tn = 2^(n+1) - 1

Tn = 1<<(n+1) - 1
Tk = 1<<(n/2 + 1) - 1
Tk = 1<<((n>>1)+1) - 1   // even cycles

Tk = 1<<((n>>1)+1) - 1<<n%2  // odd cycles
Which is the answer

Test:
java UtopianTree 
5
1
>2
2
>3
3
>6
4
>7
5
>14



*/

import java.util.Scanner;

public class UtopianTree {

    static int utopianTree(int n) {
        return (1 << ((n >> 1) + 1)) - 1 << n % 2;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int T = sc.nextInt();
        sc.skip("(\r\n|[\n\r\u2028\u2029\u0085])?");
 
        for (int i = 0; i < T; i++) {
            int cycle = sc.nextInt();
                    sc.skip("(\r\n|[\n\r\u2028\u2029\u0085])?");
            System.out.println(">" + utopianTree(cycle));
        }
        sc.close();
    }
}
