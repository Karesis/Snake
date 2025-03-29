CC = gcc
CFLAGS = -Wall -Wextra -O2 -fopenmp -I./include
LDFLAGS = -lm

SRCS = src/tensor.c src/autograd.c src/nn.c src/optim.c src/utils.c
OBJS = $(SRCS:.c=.o)
TARGET = libtensor.a

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET)
	$(CC) $(CFLAGS) test/test_tensor.c -o test_tensor $(TARGET) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) test_tensor 