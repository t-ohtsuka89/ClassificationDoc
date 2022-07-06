from datamodule import MyDataModule


def main():
    data_module = MyDataModule(
        text_dir="data/texts",
        label_dir="data/label_level1",
        batch_size=32,
        seed=0,
        add_special_token=False,
        n_truncation=None,
    )
    data_module.setup(stage="fit")
    print(data_module.vocab_size)
    print(data_module.output_size)


if __name__ == "__main__":
    main()
