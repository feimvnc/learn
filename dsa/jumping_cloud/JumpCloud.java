/*
Requirement, can jump on 0, not 1
Input: [0,1,0,0,0,1,0]  # 0 index
Output: 3 (0->2->4->6)  # index
Return minimun steps

*/

import java.util.Scanner;

public class JumpCloud {
    
    // v1
    static int jumpCloudv1(int[] c) {
        int len = c.length;
        int count = -1;

        for (int i = 0; i < len; ) {
            if (i + 1 < len && c[ i + 2] == 0) {
                i = i + 2;
            } else {
                i++;
            }
            count++;
        }
        return count;
    }

    static int jumpCloudv2(int[] c) {
        int len = c.length;
        int count = -1;

        for (int i = 0; i < len; ) {
            if (i + 2 < len && c[i + 2] == 0) {
                i++;
            }
            i++;
            count++;
        }
        return count;
    }

    // v3
    static int jumpCloudv3(int[] c) {
        int len = c.length;
        int count = -1;

        for (int i = 0; i < len; i ++, count++) {
            if (i + 2 < len && c[i + 2] == 0) {
                i++;
            }
        }
        return count;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int c[] = new int[n];
        int res = JumpCloud.jumpCloudv3(c);
        System.out.println(res);

    }
}
