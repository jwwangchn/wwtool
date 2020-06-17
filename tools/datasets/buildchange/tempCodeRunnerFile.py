plt.subplot(131), plt.imshow(lmap)
        plt.subplot(132), plt.imshow(image)
        for i0, i1 in Lpos:
            plt.scatter(junc[i0][1] * 4, junc[i0][0] * 4)
            plt.scatter(junc[i1][1] * 4, junc[i1][0] * 4)
            plt.plot([junc[i0][1] * 4, junc[i1][1] * 4], [junc[i0][0] * 4, junc[i1][0] * 4])
        plt.subplot(133), plt.imshow(lmap)
        for i0, i1 in Lneg[:150]:
            plt.plot([junc[i0][1], junc[i1][1]], [junc[i0][0], junc[i1][0]])
        plt.show()