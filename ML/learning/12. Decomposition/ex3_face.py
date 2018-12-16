from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import numpy as np

def show_image(images):
    N = 2
    M = 5
    fig = plt.figure(figsize=(10,5))
    plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
    for i in range(N):
        for j in range(M):
            k = i*M + j
            ax = fig.add_subplot(N, M, k+1)
            # 이미지 객체를 가지고 그릴때
            # ax.imshow(images[k], cmap=plt.cm.bone)
            ax.imshow(images[k].reshape(64,64), cmap=plt.cm.bone)
            ax.grid(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
    plt.suptitle('face picture')
    plt.tight_layout()
    plt.show()


def ex3_face():
    face_all = fetch_olivetti_faces()
    # faces = face_all.images[face_all.target == 20]    # 이미지 객체 불러오기
    # faces = face_all.data[face_all.target == 10]
    imageFile='./pizza2.jpg'
    imageFile2='./lena.jpg'
    
    # img1 = cv2.imread(imageFile, 0)    
    # img2 = cv2.imread(imageFile2, 0)
    # img1 = img1.reshape(262144,)
    # img2 = img2.reshape(262144,)


    img1_ori = cv2.imread(imageFile)    
    img2_ori = cv2.imread(imageFile2)
    print('img1_ori:',img1_ori.shape)
    print('img2_ori:',img2_ori.shape)

    img1 = img1_ori.reshape(img1_ori.shape[0]*img1_ori.shape[1]*img1_ori.shape[2],)
    img2 = img2_ori.reshape(img2_ori.shape[0]*img2_ori.shape[1]*img2_ori.shape[2],)
    


    faces= np.array([img1,img2])
    
    faces.shape
    size = faces.shape
    print('faces.shape:',faces.shape) # 10 * 4096
    print('faces:',faces)
    # show_image(faces)

    # 차원축소
    pca1 = PCA(n_components=2)
    W1 = pca1.fit_transform(faces)
    print('w1.shape',W1.shape)
    print(pca1.explained_variance_ratio_)
    # print(type(W1))
    # print(W1.shape)
    # faces2 = pca1.inverse_transform(W1)
    # print(type(faces2))
    # print(faces2.shape)
    # show_image(faces2)

    # 각 얼굴로부터 얻은 평준화된 데이터
    print('pca1:', pca1)
    print('pca1_mean:',pca1.mean_.shape)    # 4096
    #3차원은 그림이 그려지지 않는다.
    face_mean = pca1.mean_.reshape(img1_ori.shape[0],img1_ori.shape[1],img1_ori.shape[2])  # 2차원 데이터의 평균값
    face_p1 = pca1.components_[0].reshape(img1_ori.shape[0],img1_ori.shape[1],img1_ori.shape[2])  #
    face_p2 = pca1.components_[1].reshape(img1_ori.shape[0],img1_ori.shape[1],img1_ori.shape[2])  #
    print('face mean:',face_mean)
    print('face_p1:', face_p1)
    print('face_p2:', face_p2)
    
    plt.subplot(131)
    plt.imshow(face_mean, cmap=plt.cm.bone)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title('mean')
    '''
    plt.subplot(132)
    plt.imshow(face_p1, cmap=plt.cm.bone)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title('p1')

    plt.subplot(133)
    plt.imshow(face_p2, cmap=plt.cm.bone)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title('p2')

    plt.show()
    '''
    # 평균 얼굴에 주성분 1을 더한 사진
    N = 2
    M = 5
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(face_mean)
#     plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
#     for i in range(N):  # 0,1
#         for j in range(M):  # 0~4
#             k = i * M + j  # k: 0,1,2,3,4 5,6,7,8,9
#             ax = fig.add_subplot(N, M, k + 1)

#             w = k - 5 if k < 5 else k - 4
#             # w = -5 -4 -3 -2 -1 1 2 3 4 5
#             ax.imshow(face_mean + w * face_p1.reshape(64, 64), cmap=plt.cm.bone)
#             ax.grid(False)
#             ax.xaxis.set_ticks([])
#             ax.yaxis.set_ticks([])
    plt.suptitle('face picture')
    plt.tight_layout()
    plt.show()
ex3_face()